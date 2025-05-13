from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm


prompt_template = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Instruction:
Given a detailed description of a programming challenge, your task is to analyze the quesiton and identify the necessary knowledge points and skills required to solve the problem.
All of the potential knowledge points and skills includes:%s
Your response should clearly select which of these points of programming knowledge and skills from the list are necessary for solving this specific challenge.
Question: %s
Knowledge points and skills: <|im_end|>
<|im_start|>assistant
String, Greedy, Array, Math, Two Pointers<|im_end|>'''
# 动态候选Token生成函数
def get_dynamic_candidates(context_tokens, label_tokens, tokenizer):
    """
    根据当前上下文，动态调整可能的候选Token集合。
    """
    candidates = set()
    context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)
    for label in label_tokens:
        label_text = tokenizer.decode(label, skip_special_tokens=True)
        if label_text.startswith(context_text):
            next_token_id = label[len(context_tokens)] if len(context_tokens) < len(label) else None
            if next_token_id is not None:
                candidates.add(next_token_id)
    return list(candidates)

def llm_decode(model, tokenizer, input_ids, max_length=50):
    """
    基于生成式解码的多标签分类解码函数。
    """
    output_ids = input_ids.clone()  # 复制输入

    # 禁用梯度计算以节省显存
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(output_ids)
            logits = outputs.logits[:, -1, :]  # 获取最后一个Token的Logits

            # 采样或选取概率最高的Token
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            output_ids = torch.cat([output_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break
    
    outputs_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()


    return outputs_text

# 定义动态约束解码函数
def constrained_decoding(model, tokenizer, input_ids, label_tokens, max_length=50, separator=", "):
    """
    基于动态候选Token的多标签分类解码函数。
    """
    output_ids = input_ids.clone()  # 复制输入
    separator_id = tokenizer.encode(separator, add_special_tokens=False)  # 分隔符Token
    generated_labels = []  # 用于存储生成的标签

    # 禁用梯度计算以节省显存
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(output_ids)
            logits = outputs.logits[:, -1, :]  # 获取最后一个Token的Logits

            # 根据当前上下文动态获取可能的候选Token
            current_tokens = output_ids[0].tolist()
            dynamic_candidates = get_dynamic_candidates(current_tokens[len(input_ids[0]):], label_tokens, tokenizer)
            # print(tokenizer.decode(output_ids[0]), tokenizer.decode(dynamic_candidates))

            # 如果动态候选为空，则插入分隔符
            if not dynamic_candidates:
                dynamic_candidates = separator_id

            # 构造动态候选Token的Mask
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[:, dynamic_candidates] = False  # 允许动态候选Token
            logits.masked_fill_(mask, float('-inf'))

            # 采样或选取概率最高的Token
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            output_ids = torch.cat([output_ids, next_token], dim=-1)

            # 检查是否生成了分隔符或结束符
            if next_token.item() in separator_id:
                label_text = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True).strip()
                if label_text and label_text not in generated_labels:
                    generated_labels.append(label_text)
                # 重置当前标签解码上下文，继续生成下一个标签
                output_ids = input_ids.clone()

            if next_token.item() == tokenizer.eos_token_id:
                break

    return generated_labels

# 定义批量分类处理函数
def batch_classification(input_texts, model, tokenizer, label_tokens, max_new_tokens=50, separator=", ", consrtained_decoding=True):
    # 构造每条输入的Prompt
    prompts = input_texts

    # Tokenize批量输入
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # 禁用梯度计算以节省显存
    batch_results = []
    with torch.no_grad():
        for input_ids in tqdm(model_inputs.input_ids):
            if consrtained_decoding:
                generated_labels = constrained_decoding(
                    model, tokenizer, input_ids.unsqueeze(0), label_tokens, max_length=max_new_tokens, separator=separator
                )
                batch_results.append(separator.join(generated_labels))
            else:
                generated_labels = llm_decode(model, tokenizer, input_ids.unsqueeze(0), max_length=max_new_tokens)
                batch_results.append(generated_labels)
                # print(generated_labels)

    return batch_results

def llm_constraint_decoding_cls(model_name_or_path, dataset, topk, data_name, consrtained_decoding):
    data_len = dataset.__len__()
    labels = dataset.label_list
    test_data, topk_data = dataset.test_data, dataset.topk_data

    # if data_name == 'newcoder':
    #     prompt = open('llm_constraint_decoding/prompt_zh.txt', 'r', encoding='utf-8').read()
    #     prompt = prompt%('，'.join(labels))
    #     topk_texts = []
    #     for i in range(data_len):
    #         input_text = []
    #         for j in range(topk):
    #             input_text.append(f'题目：{topk_data[i][j][0]}\n\n知识点和技能：{topk_data[i][j][1]}')
    #         topk_texts.append(input_text)
    #     test_data = [f'题目：{ques}\n知识点和技能：' for ques in test_data]
    # el
    if data_name == 'math':
        print(data_name)
        prompt = open('llm_constraint_decoding/prompt_math.txt', 'r', encoding='utf-8').read()
        prompt = prompt%('，'.join(labels))
        topk_texts = []
        for i in range(data_len):
            input_text = []
            for j in range(topk):
                input_text.append(f'题目：{topk_data[i][j][0]}\n\n知识点和技能：{topk_data[i][j][1]}')
            topk_texts.append(input_text)
        test_data = [f'题目：{ques}\n知识点和技能：' for ques in test_data]
    else:
        prompt = open('llm_constraint_decoding/prompt_en.txt', 'r', encoding='utf-8').read()
        prompt = prompt%(', '.join(labels))
        topk_texts = []
        for i in range(data_len):
            input_text = []
            for j in range(topk):
                input_text.append(f'<|im_start|>user\nQuestion: {topk_data[i][j][0]}\nKnowledge points and skills: <|im_end|>\n<|im_start|>assistant\n{topk_data[i][j][1]}<|im_end|>')
            topk_texts.append(input_text)
        test_data = [f'Question: {ques}\nKnowledge points and skills:' for ques in test_data]
    '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Instruction:
Given a detailed description of a programming challenge, your task is to analyze the quesiton and identify the necessary knowledge points and skills required to solve the problem.
All of the potential knowledge points and skills includes:Prefix Sum, Heap (Priority Queue), Stack, Tree, Binary Tree, Segment Tree, Counting, Data Stream, Recursion, Iterator, Ordered Set, Monotonic Queue, Randomized, Sliding Window, Radix Sort, Memoization, Breadth-First Search, Number Theory, String, Brainteaser, Probability and Statistics, Bitmask, Interactive, Rejection Sampling, Strongly Connected Component, Counting Sort, Trie, Binary Search, Queue, Depth-First Search, Minimum Spanning Tree, Doubly-Linked List, Math, Shortest Path, Biconnected Component, Line Sweep, Geometry, Game Theory, Combinatorics, Enumeration, Binary Search Tree, Suffix Array, Bucket Sort, Concurrency, Binary Indexed Tree, Two Pointers, Dynamic Programming, Merge Sort, Quickselect, Graph, Hash Table, Design, Array, Database, Reservoir Sampling, Greedy
Your response should clearly select which of these points of programming knowledge and skills from the list are necessary for solving this specific challenge.
Question: DI String Match A permutation perm of n + 1 integers of all the integers in the range [0, n] can be represented as a string s of length n where:s[i] == I if perm[i]  perm[i + 1], ands[i] == D if perm[i]  perm[i + 1].Given a string s, reconstruct the permutation perm and return it. If there are multiple valid permutations perm, return any of them.
Knowledge points and skills: <|im_end|>
<|im_start|>assistant
String, Greedy, Array, Math, Two Pointers<|im_end|>'''
    input_texts = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n' + prompt + '\n' + test_data[i] + ' <|im_end|>\n<|im_start|>assistant\n' for i in range(data_len)]
    # input_texts = [f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n' + prompt for i in range(data_len)]
    print(input_texts[0])
    print(">>>>>>>>>>>>>>>>>>>>>")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    label_set = dataset.label_list
    label_separator = "，"

    label_tokens = [tokenizer.encode(label, add_special_tokens=False) for label in label_set]
    batch_results = batch_classification(input_texts, model, tokenizer, label_tokens, max_new_tokens=50, separator=label_separator, consrtained_decoding=consrtained_decoding)
    return batch_results 

if __name__ == "__main__":
    # 前缀模板
    prefix_template = "分类结果: "
    # 加载模型和Tokenizer
    model_name = "/public/home/shaoyifeng/models/qwens/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="float16",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 定义标签集合和分隔符
    label_set = ["标签1", "标签2", "标签3", "标签4"]  # 替换为你的实际标签集合
    label_separator = ", "  # 标签之间的分隔符
    label_tokens = [tokenizer.encode(label, add_special_tokens=False) for label in label_set]

    # 输入文本批处理
    input_texts = [
        "这是一段新闻文本1。",
        "这是关于科技的文本2。",
        "这是一篇关于体育的文本3。"
    ]  # 批处理的输入文本
    # 执行批量分类
    batch_results = batch_classification(input_texts, model, tokenizer, label_tokens)

    # 输出结果
    for i, (input_text, result) in enumerate(zip(input_texts, batch_results)):
        print(f"输入文本 {i+1}: {input_text}")
        print(f"分类结果 {i+1}: {result}\n")
