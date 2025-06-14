import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse
from eagle.model.ea_model import EaModel
import torch
from fastchat.model import get_conversation_template
import json
from tqdm import tqdm

def truncate_list(lst, num):
    if num not in lst:
        return lst
    first_index = lst.index(num)
    return lst[:first_index + 1]

def generate(query, temperature = 0, top_p = 0):
    assert args.model_type == "llama-2-chat" or "vicuna"
    conv = get_conversation_template(args.model_type)

    if args.model_type == "llama-2-chat":
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p
    elif args.model_type == "mixtral":
        conv = get_conversation_template("llama-2-chat")
        conv.system_message = ''
        conv.sep2 = "</s>"
    elif args.model_type == "llama-3-instruct":
        messages = [
            {"role": "system",
            "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]
    elif args.model_type == "ds-llama-3":
        messages = [] # ds-llama-3 no system prompt

    if args.model_type in ["llama-3-instruct", "ds-llama-3"]:
        messages.append({
            "role": "user",
            "content": query
        })
    else:
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)

    if args.model_type in ["llama-3-instruct", "ds-llama-3"]:
        prompt = model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = conv.get_prompt()

    if args.model_type == "llama-2-chat":
        prompt += " "

    input_ids = model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    input_len = input_ids.shape[1]
    cu_len = input_len
    total_ids=0
    final_ids = None
    eagle_ids = []
    for output_ids in model.ea_generate(input_ids, temperature=temperature, top_p=top_p,
                                        max_new_tokens=args.max_new_token,is_llama3=(args.model_type in ["llama-3-instruct", "ds-llama-3"])):
        total_ids+=1
        decode_ids = output_ids[0, cu_len:].tolist()
        decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
        if args.model_type in ["llama-3-instruct", "ds-llama-3"]:
            decode_ids = truncate_list(decode_ids, model.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        eagle_ids.append(decode_ids)
        # text = model.tokenizer.decode(decode_ids, skip_special_tokens=True, spaces_between_special_tokens=False,
        #                                 clean_up_tokenization_spaces=True, )
        cu_len = output_ids.shape[1]
        final_ids = output_ids

    final_text = model.tokenizer.decode(final_ids[0, input_len:].tolist(), skip_special_tokens=True, spaces_between_special_tokens=False,
                                        clean_up_tokenization_spaces=True, )
    # print(final_text)
    return eagle_ids, final_text


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ea-model-path",
    type=str,
    default="/home/yanxing/data/models/EAGLE3-Vicuna1.3-13B",
    help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
)
parser.add_argument("--base-model-path", type=str, default="/home/yanxing/data/models/vicuna-13b-v1.3",
                    help="path of basemodel, huggingface project or local path")
parser.add_argument(
    "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
)
parser.add_argument(
    "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
)
parser.add_argument(
    "--no-eagle3", action="store_true", help=" Not use EAGLE-3"
)
parser.add_argument("--model-type", type=str, default="vicuna",choices=["llama-2-chat","vicuna","mixtral","llama-3-instruct","ds-llama-3"])
parser.add_argument(
    "--total-token",
    type=int,
    default=60,
    help="The maximum number of new generated tokens.",
)
parser.add_argument(
    "--max-new-token",
    type=int,
    default=4096,
    help="The maximum number of new generated tokens.",
)
args = parser.parse_args()

model = EaModel.from_pretrained(
    base_model_path=args.base_model_path,
    ea_model_path=args.ea_model_path,
    total_token=args.total_token,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    load_in_4bit=args.load_in_4bit,
    load_in_8bit=args.load_in_8bit,
    device_map="auto",
    use_eagle3=(not args.no_eagle3),
)
model.eval()
# warmup(model)

# load prompt from json
all_prompts_data = {}
with open('./data/prompt.json', 'r') as f:
    all_prompts_data = json.load(f)

results_data = {}
total_prompts = sum(len(prompts_list) for prompts_list in all_prompts_data.values())
with tqdm(total=total_prompts, desc="Processing Prompts", unit="prompt", leave=True) as pbar:
    for lang_category, prompts_list in all_prompts_data.items():
        results_data[lang_category] = []
        for prompt_obj in prompts_list:
            prompt_id = prompt_obj["id"]
            prompt_text = prompt_obj["text"]
            eagle_ids, answer = generate(prompt_text)
            processed_prompt_obj = {
                "id": prompt_id,
                "user": prompt_text,
                "assistant": answer,
                "eagle_ids": eagle_ids,
                "num_eagle_ids": len(eagle_ids)
            }
            results_data[lang_category].append(processed_prompt_obj)
            pbar.update(1)

type_to_name = {
    'llama-2-chat': 'llama2',
    'vicuna': 'vicuna',
    'mixtral': 'mixtral',
    'llama-3-instruct': 'llama3',
    'ds-llama-3': 'deepseek'
}

outfile = f'./data/{type_to_name[args.model_type]}.json'
with open(outfile, 'w', encoding='utf-8') as f:
    json.dump(results_data, f, indent=4, ensure_ascii=False)