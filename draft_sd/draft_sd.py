import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import torch.nn.functional as F
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# --- 全局调试开关 ---
DEBUG_MODE = True # 设置为 True 开启调试信息打印，设置为 False 关闭

# --- 1. 模型加载 (作为全局变量，在应用启动时加载) ---
model_main = None
tokenizer_main = None
model_draft = None
tokenizer_draft = None
main_model_dtype = None
draft_model_dtype = None

# 定义模型路径
# 请根据您的实际模型路径进行修改
MAIN_MODEL_PATH = "/home/yanxing/data/models/Qwen3-4B"
DRAFT_MODEL_PATH = "/home/yanxing/data/models/Qwen3-0.6B"


def print_tensor_summary(tensor, name):
    # 将张量移动到CPU并转换为float32以便计算
    tensor_cpu = tensor.detach().to(torch.float32).cpu()
    mean = tensor_cpu.mean().item()
    std = tensor_cpu.std().item()
    abs_sum = tensor_cpu.abs().sum().item()
    # 打印几个样本值，从张量中间取一个
    sample_val = tensor_cpu.flatten()[tensor_cpu.numel() // 2].item()
    print(f"  {name}:")
    print(f"    - Mean: {mean:.4f}, Std: {std:.4f}, AbsSum: {abs_sum:.4f}, Sample: {sample_val:.4f}")

def print_dynamic_cache_info(
    cache: "DynamicCache",
    cache_name: str = "Cache",
    layer_to_inspect: int = 0,
    title: str = None
):
    if title:
        header = f"--- {title} ---"
    else:
        header = f"--- {cache_name} Info ---"

    print(header)

    if cache is None:
        print("Cache is None.")
        print("-" * (len(header)) + "\n")
        return

    num_layers = len(cache)
    if not (0 <= layer_to_inspect < num_layers):
        print(f"Invalid layer_to_inspect: {layer_to_inspect}. Cache has {num_layers} layers.")
        print("-" * (len(header)) + "\n")
        return

    # --- 获取要检查的层的 Key/Value 张量 ---
    key, val = cache[layer_to_inspect]
    seq_len = key.shape[2]

    # --- 打印基本信息 ---
    print(f"Inspecting Layer: {layer_to_inspect}")
    print(f"Sequence Length: {seq_len}")
    print(f"Key Shape:   {key.shape}")
    print(f"Value Shape: {val.shape}")
    print_tensor_summary(key, "Key Cache (Full)")
    print_tensor_summary(val, "Value Cache (Full)")
    print("-" * (len(header)) + "\n")

def format_token_display(token_text):
    """
    将token文本中的特殊字符转换为更直观的显示形式
    """
    if not token_text:
        return "''"
    # 特殊字符映射表
    special_chars = {
        '\n': '\\n',      # 换行
        '\r': '\\r',      # 回车
        '\t': '\\t',      # 制表符
        '\b': '\\b',      # 退格
        '\f': '\\f',      # 换页
        '\v': '\\v',      # 垂直制表符
        '\a': '\\a',      # 响铃
        '\0': '\\0',      # 空字符
        '\\': '\\\\',     # 反斜杠
    }
    # 替换特殊字符
    display_text = token_text
    for char, replacement in special_chars.items():
        display_text = display_text.replace(char, replacement)
    # 处理其他不可见字符 (ASCII 0-31 和 127)
    result = ""
    for char in display_text:
        char_code = ord(char)
        if char_code < 32 or char_code == 127:
            # 对于其他控制字符，显示为 \x格式
            if char not in special_chars:
                result += f"\\x{char_code:02x}"
            else:
                result += char
        else:
            result += char
    return result

def load_model(model_path):
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=model_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit(1)
    return model, tokenizer

class TreeNode:
    """
    表示生成树中的一个节点。
    Attributes:
        token_id (int): 该节点代表的 token ID。
        log_prob (float): 从父节点生成此节点的条件对数概率。
        cumulative_log_prob (float): 从根到此节点的路径的累积对数概率。
        parent (TreeNode): 父节点对象，根节点的父节点为 None。
        children (list[TreeNode]): 子节点对象列表。
        depth (int): 节点在树中的深度 (根为0)。
        node_id (int): 在树中创建的唯一顺序ID，用于最终排序。
    """
    def __init__(self, token_id, log_prob, parent, node_id):
        self.token_id = token_id
        self.log_prob = log_prob
        self.parent = parent
        self.node_id = node_id # 唯一的、按创建顺序的ID
        self.children = []

        if parent:
            self.depth = parent.depth + 1
            self.cumulative_log_prob = parent.cumulative_log_prob + log_prob
            parent.children.append(self)
        else:
            # 适用于连接到虚拟根节点的首层节点
            self.depth = 0
            self.cumulative_log_prob = log_prob

class Tree:
    """
    使用 TreeNode 对象管理草稿 token 树的生长、剪枝和定稿。
    """
    def __init__(self, top_k: int, prompt_len: int, device: torch.device):
        self.top_k = top_k
        self.prompt_len = prompt_len
        self.device = device
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        # build a virtual root node
        self.root = TreeNode(token_id=-1, log_prob=0.0, parent=None, node_id=-1)
        # 当前待扩展的叶子节点
        self.active_leaves = []
        self._node_counter = 0
        # 用于高效生成 attention mask 的内部状态 (与上一版相同)
        self._inference_mask = torch.eye(self.top_k, device=self.device)[None, None]

    def _get_all_nodes_bfs(self):
        """使用广度优先搜索返回树中所有（非虚拟根）节点。"""
        all_nodes = []
        queue = list(self.root.children)
        while queue:
            node = queue.pop(0)
            all_nodes.append(node)
            queue.extend(node.children)
        return all_nodes

    def add_initial_level(self, logits: torch.Tensor):
        log_probs = self.logsoftmax(logits)
        top_k_results = torch.topk(log_probs, self.top_k, dim=-1)
        initial_tokens = top_k_results.indices.squeeze(0)
        initial_scores = top_k_results.values.squeeze(0)

        for i in range(self.top_k):
            node = TreeNode(
                token_id=initial_tokens[i].item(),
                log_prob=initial_scores[i].item(),
                parent=self.root,
                node_id=self._node_counter
            )
            self._node_counter += 1
            self.active_leaves.append(node)

    def grow(self, logits: torch.Tensor):
        log_probs = self.logsoftmax(logits)
        top_k = torch.topk(log_probs, self.top_k, dim=-1)
        topk_tokens, topk_log_probs = top_k.indices, top_k.values
        candidates = []
        # 1. 为每个活跃叶子节点生成所有可能的子节点
        for i, parent_leaf in enumerate(self.active_leaves):
            for j in range(self.top_k):
                child_node = TreeNode(
                    token_id=topk_tokens[i, j].item(),
                    log_prob=topk_log_probs[i, j].item(),
                    parent=parent_leaf,
                    node_id=self._node_counter
                )
                self._node_counter += 1
                candidates.append(child_node)

        # 2. 剪枝：根据累积概率对所有新生成的候选节点进行排序
        candidates.sort(key=lambda n: n.cumulative_log_prob, reverse=True)
        # 3. 选出 top_k 个最好的节点作为下一轮的活跃叶子
        new_active_leaves = candidates[:self.top_k]
        # 4. 更新 attention mask 的内部状态
        #    找出新叶子的父节点在上一轮 active_leaves 中的索引
        parent_map = {leaf: i for i, leaf in enumerate(self.active_leaves)}
        parent_remap_indices = [parent_map[node.parent] for node in new_active_leaves]
        parent_remap_indices = torch.tensor(parent_remap_indices, device=self.device)
        self._inference_mask = torch.cat(
            (self._inference_mask[:, :, parent_remap_indices],
             torch.eye(self.top_k, device=self.device)[None, None]),
            dim=3
        )
        # 5. 更新活跃叶子列表
        self.active_leaves = new_active_leaves

    def get_inference_inputs(self):
        """
        获取当前轮次模型推理所需的 input_ids, attention_mask, position_ids。
        """
        if not self.active_leaves:
            raise ValueError("Tree has no active leaves to grow from.")
        # 从活跃叶子节点对象中提取信息，组装成张量
        input_ids = torch.tensor([[node.token_id for node in self.active_leaves]], device=self.device)
        # 所有活跃叶子都在同一深度
        current_depth = self.active_leaves[0].depth - 1
        position_ids = torch.full_like(input_ids, self.prompt_len + current_depth, device=self.device)
        # 生成 attention mask
        left_mask = torch.zeros([1, 1, self.top_k, self.prompt_len], dtype=torch.float32, device=self.device)
        right_mask = (1 - self._inference_mask) * torch.finfo(torch.float32).min
        attention_mask = torch.cat([left_mask, right_mask], dim=-1).to(torch.bfloat16)
        return input_ids, attention_mask, position_ids

    def finalize(self, max_draft_tokens: int):
        """
        从整棵树中选出最终的草稿，并生成 verifier 所需的 mask 和 position_ids。
        """
        # 1. 获取所有节点，按分数排序，选出最终草稿节点
        all_nodes = self._get_all_nodes_bfs()
        all_nodes.sort(key=lambda n: n.cumulative_log_prob, reverse=True)
        draft_nodes = all_nodes[:max_draft_tokens]
        # 2. 按创建顺序 (node_id) 排序，以确保父节点总在子节点之前
        draft_nodes.sort(key=lambda n: n.node_id)
        # 3. 生成 verifier 所需的输入
        draft_tokens = torch.tensor([[node.token_id for node in draft_nodes]], device=self.device)
        tree_position_ids = torch.tensor([node.depth - 1 for node in draft_nodes], device=self.device)
        # 4. 构建 tree_mask
        node_id_to_idx = {node.node_id: i for i, node in enumerate(draft_nodes)}
        tree_mask = torch.eye(max_draft_tokens, device=self.device, dtype=torch.bool)
        for i, node in enumerate(draft_nodes):
            if node.parent and node.parent.node_id in node_id_to_idx:
                parent_idx = node_id_to_idx[node.parent.node_id]
                # 每个节点的 attention mask 是其父节点的 mask 加上它自己
                tree_mask[i].add_(tree_mask[parent_idx])
        tree_mask = tree_mask.float()[None, None]

        # 5. 构建 retrieve_indices
        leaf_nodes = [node for node in draft_nodes if not any(child in draft_nodes for child in node.children)]
        max_depth = 0
        if draft_nodes:
            max_depth = max(node.depth for node in draft_nodes)
        paths = []
        for leaf in leaf_nodes:
            path = []
            curr = leaf
            while curr.parent:
                path.append(node_id_to_idx[curr.node_id])
                curr = curr.parent
            paths.append(list(reversed(path)))
        retrieve_indices = torch.full((len(paths), max_depth), -1, dtype=torch.long, device=self.device)
        for i, path in enumerate(paths):
            retrieve_indices[i, :len(path)] = torch.tensor(path, dtype=torch.long, device=self.device)
        return draft_tokens, tree_mask, tree_position_ids, retrieve_indices

    def __repr__(self):
        if not self.root.children:
            return "<Tree (empty)>"
        lines = ["Tree Structure (token, cumulative_log_prob):"]
        lines.append(self._build_repr_recursive(self.root, ""))
        return "\n".join(lines)

    def _build_repr_recursive(self, node, prefix):
        lines = []
        children = node.children
        for i, child in enumerate(children):
            is_last = (i == len(children) - 1)
            connector = "└── " if is_last else "├── "
            token_str = format_token_display(self.tokenizer.decode([child.token_id]))
            node_info = f"'{token_str}' ({child.token_id}, {child.cumulative_log_prob:.2f})"
            lines.append(f"{prefix}{connector}{node_info}")
            new_prefix = prefix + ("    " if is_last else "│   ")
            child_repr = self._build_repr_recursive(child, new_prefix)
            if child_repr:
                lines.append(child_repr)
        return "\n".join(lines)

class DraftModel:
    def __init__(self, model_path):
        self.model, self.tokenizer = load_model(model_path)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.depth = 8
        self.top_k = 5
        self.max_draft_tokens = self.depth * self.top_k
        self.past_key_values = None
        self.past_len = 0

    @torch.no_grad()
    def update_kv_cache(self):
        if self.past_key_values is None:
            return
        for layer_idx in range(len(self.past_key_values)):
            self.past_key_values.key_cache[layer_idx] = self.past_key_values.key_cache[layer_idx][:, :, :self.past_len, :]
            self.past_key_values.value_cache[layer_idx] = self.past_key_values.value_cache[layer_idx][:, :, :self.past_len, :]
        self.past_key_values._seen_tokens = self.past_len

    @torch.no_grad()
    def generate(self, input_ids):
        self.update_kv_cache()
        input_ids = input_ids.to(self.model.device)
        self.past_len += input_ids.shape[1]
        tree = Tree(top_k=self.top_k, prompt_len=self.past_len, device=self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids, past_key_values=self.past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]
            self.past_key_values = outputs.past_key_values

        tree.add_initial_level(logits)
        for _ in range(self.depth):
            input_ids, attention_mask, position_ids = tree.get_inference_inputs()
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=self.past_key_values,
                    use_cache=True
                )
            self.past_key_values = outputs.past_key_values
            tree.grow(outputs.logits[0])
        draft_tokens, tree_mask, tree_position_ids, retrieve_indices = tree.finalize(self.max_draft_tokens)
        if DEBUG_MODE and False:
            tree.tokenizer = self.tokenizer
            print(tree)
            print("\n--- Final Draft ---")
            print(f"Draft Tokens: {draft_tokens}")
            print(f"Tree Mask Shape: {tree_mask.shape}")
            print(f"Tree Position IDs: {tree_position_ids}")
            print(f"Retrieve Indices:\n{retrieve_indices}")
        return draft_tokens, tree_mask, tree_position_ids, retrieve_indices

class MainModel:
    def __init__(self, model_path):
        self.model, self.tokenizer = load_model(model_path)
        self.past_len = 0
        self.past_key_values = None
        self.tokens = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def gen_position_ids(self, prompt_len, draft_position_ids):
        position_ids = torch.arange(prompt_len, device=self.device)
        position_ids = torch.concat([position_ids, draft_position_ids + prompt_len], dim=0)
        position_ids = position_ids.unsqueeze(0) + self.past_len
        return position_ids

    def gen_attention_mask(self, prompt_len, draft_attention_mask):
        '''
        [1, 0, 0, 0, 0]  [0, 0, 0]
        [1, 1, 0, 0, 0]  [0, 0, 0]
        [1, 1, 1, 0, 0]  [0, 0, 0]
        [1, 1, 1, 1, 0]  [0, 0, 0]
        [1, 1, 1, 1, 1]  [0, 0, 0]

        [1, 1, 1, 1, 1]  [ draft ]
        [1, 1, 1, 1, 1]  [  pos  ]
        [1, 1, 1, 1, 1]  [  ids  ]
        '''
        draft_len = draft_attention_mask.shape[-1]
        up_left_mask = torch.tril(torch.ones([1, 1, prompt_len, prompt_len], device=self.device))
        up_right_mask = torch.zeros([1, 1, prompt_len, draft_len], device=self.device)
        up_mask = torch.concat([up_left_mask, up_right_mask], dim=-1)
        down_left = torch.ones([1, 1, draft_len, prompt_len], device=self.device)
        down_right = draft_attention_mask.to(self.device)
        down_mask = torch.concat([down_left, down_right], dim=-1)
        attention_mask = torch.concat([up_mask, down_mask], dim=-2)
        if self.past_len > 0:
            past_mask = torch.ones([1, 1, prompt_len + draft_len, self.past_len], device=self.device)
            attention_mask = torch.concat([past_mask, attention_mask], dim=-1)
        attention_mask = (1 - attention_mask) * torch.finfo(torch.float32).min
        return attention_mask.to(torch.bfloat16)

    @torch.no_grad()
    def update_kv_cache(self,
        accepted_indices,
        prompt_len,
        accept_length,
    ):
        for layer_idx in range(len(self.past_key_values)):
            key_cache, value_cache = self.past_key_values[layer_idx]
            device = key_cache.device
            destination_slice_key = key_cache[:, :, prompt_len : prompt_len + accept_length, :]
            destination_slice_value = value_cache[:, :, prompt_len : prompt_len + accept_length, :]
            accepted_keys = key_cache.index_select(dim=2, index=accepted_indices)
            accepted_values = value_cache.index_select(dim=2, index=accepted_indices)
            destination_slice_key.copy_(accepted_keys)
            destination_slice_value.copy_(accepted_values)
        new_seq_len = prompt_len + accept_length
        for layer_idx in range(len(self.past_key_values)):
            self.past_key_values.key_cache[layer_idx] = self.past_key_values.key_cache[layer_idx][:, :, :new_seq_len, :]
            self.past_key_values.value_cache[layer_idx] = self.past_key_values.value_cache[layer_idx][:, :, :new_seq_len, :]
        self.past_key_values._seen_tokens = new_seq_len
        self.past_len = new_seq_len.item()

    def evaluate_posterior(self,
            logits: torch.Tensor,
            candidates: torch.Tensor,
            ):
        # Find the tokens that match the maximum logits for each position in the sequence
        verify_tokens = torch.argmax(logits, dim=-1)
        posterior_mask = (candidates.to(logits.device) == verify_tokens[:, :-1]).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        accept_ids = verify_tokens[None, best_candidate, :accept_length + 1]
        return best_candidate, accept_length, accept_ids, logits[best_candidate, accept_length]

    @torch.no_grad()
    def generate(self, input_ids):
        seq_len = input_ids.shape[1]
        debug = False
        if self.past_len == 0:
            self.past_len = seq_len
            attention_mask = torch.tril(torch.ones([1, 1, seq_len, seq_len], device=self.device))
            attention_mask = (1 - attention_mask) * torch.finfo(torch.float32).min
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
        else:
            attention_mask = torch.zeros([1, 1, 1, self.past_len + 1], dtype=torch.long, device=self.device)
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0) + self.past_len
            self.past_len += 1
            # if input_ids.item() == 68805: debug = True
        if debug:
            print('\nself.past_len = ', self.past_len)
            print('position_ids = ', position_ids)
            print('attention_mask.shape = ', attention_mask.shape)
            print_dynamic_cache_info(self.past_key_values)
        outputs = self.model(
            input_ids,
            attention_mask = attention_mask.to(torch.bfloat16),
            position_ids=position_ids,
            past_key_values=self.past_key_values
        )
        logits = outputs.logits[0, -1, :]
        if debug:
            # print('\n', self.tokens)
            # print(self.tokenizer.batch_decode(torch.tensor(self.tokens)))
            print(logits)
            print(torch.topk(logits, k=5))
            exit(0)
        self.past_key_values = outputs.past_key_values
        token = torch.argmax(logits, dim=-1).reshape([1, 1])
        self.tokens.append(token.item())
        return token

    @torch.no_grad()
    def verify(self,
            draft_tokens,
            draft_position_ids,
            draft_attention_mask,
            retrieve_indices,
            prompt_ids
            ):
        prompt_len = prompt_ids.shape[1]
        input_ids = torch.concat([prompt_ids, draft_tokens], dim=1)
        position_ids = self.gen_position_ids(prompt_len, draft_position_ids)
        attention_mask = self.gen_attention_mask(prompt_len, draft_attention_mask)
        outputs = self.model(
            input_ids,
            attention_mask = attention_mask,
            position_ids=position_ids,
            past_key_values=self.past_key_values
        )
        self.past_key_values = outputs.past_key_values
        logits = outputs.logits
        # draft candidates
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(self.device)
        draft_tokens = torch.cat((draft_tokens, padding), dim=1)
        candidates = draft_tokens[0, retrieve_indices]
        # draft logits
        logits = logits[0, prompt_len - 1:, :]
        logits_retrieve_indices = F.pad(retrieve_indices + 1, pad=(1, 0, 0, 0), mode='constant', value=0)
        logits = logits[logits_retrieve_indices]
        # print(f'logits = {torch.argmax(logits, dim=-1)}, txt = {self.tokenizer.batch_decode(torch.argmax(logits, dim=-1))}')
        candidates_valid = candidates.clone()
        candidates_valid[candidates < 0] = 0
        # print(f'candidates = {candidates}, txt = {self.tokenizer.batch_decode(candidates_valid)}')
        best_candidate, accept_length, accept_ids, sample_p = self.evaluate_posterior(logits, candidates)
        accepted_indices = retrieve_indices[best_candidate, :accept_length] + self.past_len + prompt_len
        self.update_kv_cache(accepted_indices, self.past_len + prompt_len, accept_length)
        return best_candidate, accept_length, accept_ids, sample_p

class SpeculativeModel:
    def __init__(self, main_path, draft_path):
        self.main_model = MainModel(main_path)
        self.draft_model = DraftModel(draft_path)
        self.tokenizer = self.main_model.tokenizer
        self.device = self.main_model.model.device
        self.stop_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def generate(self,
                prompt,
                use_speculative=True,
                max_new_tokens=200):
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=self.device)
        prompt_ids = input_ids
        cur_new_tokens = 0
        while cur_new_tokens < max_new_tokens:
            if not use_speculative:
                input_ids = self.main_model.generate(input_ids)
                cur_new_tokens += 1
                yield input_ids
                continue
            draft_tokens, tree_mask, tree_position_ids, retrieve_indices = self.draft_model.generate(input_ids)
            best_candidate, accept_length, accept_ids, sample_p = self.main_model.verify(
                draft_tokens,
                draft_position_ids=tree_position_ids,
                draft_attention_mask=tree_mask,
                retrieve_indices=retrieve_indices,
                prompt_ids=prompt_ids,
            )
            input_ids = accept_ids
            cur_new_tokens += input_ids.shape[1]
            # print(self.tokenizer.batch_decode(input_ids))
            yield input_ids
            prompt_ids = accept_ids[:, -1:]
            if self.stop_id in input_ids.tolist():
                break
        # print(f"Average Accept Length: {avg_accept / max_new_tokens}")


def webui(spec_model):
    def webui_generate(
        prompt,
        highlight_accepted_tokens, # Keep this for UI, even if always True in simple mode
        max_new_tokens,
        max_draft_tokens,
        num_branches,
        max_depth,
        # Removed: temperature, top_p, top_k, use_speculative_decoding
    ):
        """Gradio的控制器函数，处理用户输入并调用模型生成。"""
        if spec_model is None:
            yield "<h3><font color='red'>错误：模型尚未加载，请检查控制台日志。</font></h3>", "0.00 tokens/s", "0.00"
            return

        full_text_html = ""
        total_token_count = 0
        total_llm_calls = 0
        start_time = time.time()

        try:
            generator = spec_model.generate(
                prompt=prompt,
                use_speculative = False,
                max_new_tokens=max_new_tokens,
                # max_draft_tokens=max_draft_tokens,
                # num_branches=num_branches,
                # max_depth=max_depth,
                # # Removed: temperature, top_p, top_k, use_speculative_decoding (always True implied)
            )

            for output_ids in generator:
                output_ids = output_ids.reshape(-1, 1)
                token_count = output_ids.shape[0]
                decode_token = output_ids[:1, 0]
                decode_txt = spec_model.tokenizer.batch_decode(decode_token)[0]
                full_text_html += f"{decode_txt}"
                if token_count > 1:
                    accepted_tokens = output_ids[1:, 0]
                    accepted_txt = ''.join(spec_model.tokenizer.batch_decode(accepted_tokens))
                    full_text_html += f"<span style='color: orange; font-weight: bold;'>{accepted_txt}</span>"

                total_token_count += token_count
                total_llm_calls += 1
                elapsed_time = time.time() - start_time
                speed = total_token_count / elapsed_time if elapsed_time > 0 else 0
                compression_ratio = total_token_count / total_llm_calls if total_llm_calls > 0 else 0

                yield full_text_html, f"{speed:.2f} tokens/s", f"{compression_ratio:.2f}"

        except Exception as e:
            print(f"生成过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            full_text_html += f"<br><h3><font color='red'>生成中断: {e}</font></h3>"
            yield full_text_html, "Error", "Error"
    import gradio as gr
    with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container { max-width: 1200px; margin: auto; }") as demo:
        gr.Markdown(
            """
            # 🌳 树形投机解码 WebUI
            使用 Qwen3-4B 作为主模型，Qwen3-0.6B 作为草稿模型进行树形投机解码。
            - **Speed (tokens/s)**: 每秒生成的token数量。
            - **Compression Ratio**: 生成的token总数 / 主模型(LLM)前向推理次数。该值大于1表示有加速效果。
            """
        )

        with gr.Row():
            speed_display = gr.Textbox(label="Speed", value="0.00 tokens/s", interactive=False, scale=1)
            compression_ratio_display = gr.Textbox(label="Compression Ratio", value="0.00", interactive=False, scale=1)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 设置")
                # use_speculative_decoding_checkbox = gr.Checkbox(label="✅ 启用投机解码", value=True, visible=False) # Removed for simplicity
                highlight_accepted_tokens_checkbox = gr.Checkbox(
                    label="🎨 高亮投机部分 (橙色)", value=True
                )
                prompt_input = gr.Textbox(lines=3, label="📝 输入提示词", value="请用中文写一首关于人工智能未来发展的五言绝句诗。")

                with gr.Accordion("模型参数", open=True):
                    max_new_tokens_slider = gr.Slider(minimum=10, maximum=1024, value=256, step=1, label="最大新Token数")
                    # Removed: temperature_slider, top_p_slider, top_k_slider

                with gr.Accordion("投机解码参数", open=True):
                    max_depth_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="树最大深度")
                    num_branches_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="每节点分支数 (k)")
                    max_draft_tokens_slider = gr.Slider(minimum=1, maximum=50, value=15, step=1, label="草稿最大Token数")

                with gr.Row():
                    send_btn = gr.Button("🚀 生成", variant="primary")
                    stop_btn = gr.Button("⏹️ 停止")
                    clear_btn = gr.Button("🗑️ 清空")

            with gr.Column(scale=2):
                gr.Markdown("### 💡 输出结果")
                output_text = gr.HTML(label="Output", value="")

        send_event = send_btn.click(
            fn=webui_generate,
            inputs=[
                prompt_input, highlight_accepted_tokens_checkbox, # Removed use_speculative_decoding_checkbox
                max_new_tokens_slider, max_draft_tokens_slider, num_branches_slider, max_depth_slider,
                # Removed: temperature_slider, top_p_slider, top_k_slider
            ],
            outputs=[output_text, speed_display, compression_ratio_display],
        )

        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[send_event], queue=False)

        def clear_action():
            return "", "", "0.00 tokens/s", "0.00"

        clear_btn.click(
            fn=clear_action, inputs=[],
            outputs=[prompt_input, output_text, speed_display, compression_ratio_display],
            cancels=[send_event], queue=False
        )
        demo.launch(share=False, enable_queue=True, server_name="0.0.0.0")

def simple(spec_model):
    RESET = "\033[0m"
    GREEN = "\033[32;1m"
    YELLOW = "\033[33;4m"
    prompt = "请用中文写一首关于人工智能未来发展的五言绝句诗。"
    generator = spec_model.generate(prompt, use_speculative = False, max_new_tokens = 500)
    all_tokens = 0
    decode_times = 0
    start_time = time.time()
    for i, output_ids in enumerate(generator):
        output_ids = output_ids.reshape(-1, 1)
        all_tokens += output_ids.shape[0]
        decode_times += 1
        decode_token = output_ids[:1, 0]
        accepted_tokens = output_ids[1:, 0]
        decode_txt = spec_model.tokenizer.batch_decode(decode_token)[0]
        accepted_txt = ''.join(spec_model.tokenizer.batch_decode(accepted_tokens))
        # txts = spec_model.tokenizer.batch_decode(output_ids)
        # print(''.join(txts), end="", flush=True)
        print(f"{decode_txt}{YELLOW}{accepted_txt}{RESET}", end="", flush=True)
    elapsed_time = time.time() - start_time
    print(f"\nSpeed: {all_tokens / elapsed_time} tokens/s")
    print(f"\nAverage Accept Length: {all_tokens / decode_times}")

if __name__ == "__main__":
    spec_model = SpeculativeModel(MAIN_MODEL_PATH, DRAFT_MODEL_PATH)
    simple(spec_model)
    # webui(spec_model)
