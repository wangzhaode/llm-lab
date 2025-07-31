import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from operator import itemgetter

MODEL_ID = '/Users/wangzhaode/workspace/Qwen3-Reranker-0.6B'

def qwen_rerank():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).eval()
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    max_length = 8192
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    @torch.no_grad()
    def compute_scores(instruction, query, documents):
        query_doc_pairs = [f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}" for doc in documents]
        # process_inputs
        inputs = tokenizer(
            query_doc_pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)

        print(f"Input shape: {inputs['input_ids'].shape}")

        batch_scores = model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    @torch.no_grad()
    def compute_scores_opt(instruction, query, documents):
        pair_prefix = tokenizer.encode(f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: ", add_special_tokens=False)
        products_tokens = tokenizer(
            documents, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
        )
        input_ids_list = prefix_tokens + pair_prefix
        prefix_len = len(input_ids_list)
        position_ids_list = list(range(prefix_len))
        product_spans = []
        current_len = prefix_len
        for ele in products_tokens['input_ids']:
            product_with_suffix = ele + suffix_tokens
            product_len = len(product_with_suffix)
            start_idx = current_len
            end_idx = current_len + product_len
            product_spans.append((start_idx, end_idx))
            current_len = end_idx
            input_ids_list.extend(product_with_suffix)
            position_ids_list.extend(list(range(prefix_len, prefix_len + product_len)))
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=model.device)
        position_ids = torch.tensor([position_ids_list], dtype=torch.long, device=model.device)
        total_len = len(input_ids_list)
        attention_mask = torch.tril(torch.ones(total_len, total_len, dtype=torch.long, device=model.device))
        for i in range(len(product_spans)):
            for j in range(len(product_spans)):
                if i == j:
                    continue
                start_i, end_i = product_spans[i]
                start_j, end_j = product_spans[j]
                attention_mask[start_i:end_i, start_j:end_j] = 0
        attention_mask = (1 - attention_mask) * torch.finfo(torch.float32).min
        attention_mask = attention_mask.reshape(1, 1, total_len, total_len)
        inputs = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask
        }
        logits_ids = torch.tensor([end - 1 for _, end in product_spans], dtype=torch.long, device=model.device)

        print(f"Input shape: {inputs['input_ids'].shape}")

        scores = model(**inputs).logits[0, logits_ids, :]
        true_vector = scores[:, token_true_id]
        false_vector = scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def sort_documents(documents, scores):
        indexed_scores = list(enumerate(scores))
        sorted_indexed_scores = sorted(indexed_scores, key=itemgetter(1), reverse=True)
        reranked_indices = [index for index, score in sorted_indexed_scores]
        print("\nOriginal documents:")
        for i, p in enumerate(documents):
            print(f"[index: {i}, score: {indexed_scores[i][1]:.4f}]: {p}")

        print("\nReranked indices:")
        print(reranked_indices)

        print("\nReranked documents:")
        for i in reranked_indices:
            print(documents[i])


    instruction = "Given a web search query, retrieve relevant passages that answer the query"

    query = "What is the capital of China?"
    documents = [
        "The Amazon River is the largest river by discharge volume of water in the world.",
        "China's capital city is Beijing, which hosted the 2008 Summer Olympics.",
        "The capital of Japan is Tokyo, while the capital of South Korea is Seoul.",
        "The mitochondria is the powerhouse of the cell.",
        "Nanjing was the capital of China during various historical periods, but the current capital is Beijing.",
        "The Forbidden City, a major landmark, is located in the heart of Beijing, China's capital.",
        "The capital of China is Beijing.",
        "A new study suggests that regular exercise can significantly improve cardiovascular health.",
        "Many people think Hong Kong is the capital of China, but it is a Special Administrative Region.",
        "The Great Wall of China is one of the most famous structures in the world, with sections easily accessible from Beijing.",
        "To bake a good cake, you must preheat your oven.",
        "The answer is Beijing.",
        "Shanghai is China's largest city, but the nation's capital is located in Beijing.",
        "Beijing, the capital of the People's Republic of China, is a city of immense historical significance.",
        "Capital investment in Chinese technology companies has reached record highs this year.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        "China is governed from its capital, Beijing, where the central government offices are located.",
        "China has many large cities, including Shanghai, Guangzhou, and Shenzhen.",
    ]


    start = time.time()
    scores = compute_scores(instruction, query, documents)
    end = time.time()
    print(f"Reranking took {end - start:.2f} seconds")


    start = time.time()
    scores = compute_scores_opt(instruction, query, documents)
    end = time.time()
    print(f"Opt Reranking took {end - start:.2f} seconds")

    sort_documents(documents, scores)

    # Example output:

    # Input shape: torch.Size([18, 108])
    # Reranking took 2.56 seconds
    # Input shape: torch.Size([1, 527])
    # Opt Reranking took 0.74 seconds

    # Original documents:
    # [index: 0, score: 0.0000]: The Amazon River is the largest river by discharge volume of water in the world.
    # [index: 1, score: 0.9988]: China's capital city is Beijing, which hosted the 2008 Summer Olympics.
    # [index: 2, score: 0.0033]: The capital of Japan is Tokyo, while the capital of South Korea is Seoul.
    # [index: 3, score: 0.0000]: The mitochondria is the powerhouse of the cell.
    # [index: 4, score: 0.9966]: Nanjing was the capital of China during various historical periods, but the current capital is Beijing.
    # [index: 5, score: 0.7085]: The Forbidden City, a major landmark, is located in the heart of Beijing, China's capital.
    # [index: 6, score: 0.9995]: The capital of China is Beijing.
    # [index: 7, score: 0.0000]: A new study suggests that regular exercise can significantly improve cardiovascular health.
    # [index: 8, score: 0.9725]: Many people think Hong Kong is the capital of China, but it is a Special Administrative Region.
    # [index: 9, score: 0.0277]: The Great Wall of China is one of the most famous structures in the world, with sections easily accessible from Beijing.
    # [index: 10, score: 0.0000]: To bake a good cake, you must preheat your oven.
    # [index: 11, score: 0.3938]: The answer is Beijing.
    # [index: 12, score: 0.9883]: Shanghai is China's largest city, but the nation's capital is located in Beijing.
    # [index: 13, score: 0.9950]: Beijing, the capital of the People's Republic of China, is a city of immense historical significance.
    # [index: 14, score: 0.0044]: Capital investment in Chinese technology companies has reached record highs this year.
    # [index: 15, score: 0.0000]: Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.
    # [index: 16, score: 0.9963]: China is governed from its capital, Beijing, where the central government offices are located.
    # [index: 17, score: 0.1084]: China has many large cities, including Shanghai, Guangzhou, and Shenzhen.

    # Reranked indices:
    # [6, 1, 4, 16, 13, 12, 8, 5, 11, 17, 9, 14, 2, 0, 7, 10, 3, 15]

    # Reranked documents:
    # The capital of China is Beijing.
    # China's capital city is Beijing, which hosted the 2008 Summer Olympics.
    # Nanjing was the capital of China during various historical periods, but the current capital is Beijing.
    # China is governed from its capital, Beijing, where the central government offices are located.
    # Beijing, the capital of the People's Republic of China, is a city of immense historical significance.
    # Shanghai is China's largest city, but the nation's capital is located in Beijing.
    # Many people think Hong Kong is the capital of China, but it is a Special Administrative Region.
    # The Forbidden City, a major landmark, is located in the heart of Beijing, China's capital.
    # The answer is Beijing.
    # China has many large cities, including Shanghai, Guangzhou, and Shenzhen.
    # The Great Wall of China is one of the most famous structures in the world, with sections easily accessible from Beijing.
    # Capital investment in Chinese technology companies has reached record highs this year.
    # The capital of Japan is Tokyo, while the capital of South Korea is Seoul.
    # The Amazon River is the largest river by discharge volume of water in the world.
    # A new study suggests that regular exercise can significantly improve cardiovascular health.
    # To bake a good cake, you must preheat your oven.
    # The mitochondria is the powerhouse of the cell.
    # Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.

qwen_rerank()