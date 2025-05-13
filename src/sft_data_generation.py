from load_data import FewShotDataset
import random 
import json

def load_data(data_name, fold_id, topk):
    dataset = FewShotDataset(path1=f'benchmarks/{data_name}/', fold_id=fold_id, topk=topk)
    return dataset

def sft_data_generate(dataset_list):
    instructions = open("llm_constraint_decoding/prompt_en.txt", "r").read()
    sft_data = []
    rate = [1000, 50, 100]
    # for dataset_name in dataset_list:
    for id in range(len(dataset_list)):
        dataset_name = dataset_list[id]
        dataset = load_data(dataset_name, "0", 3)
        instruction_data = []
        labels = dataset.label_list
        data, data_labels = dataset.data, dataset.label
        for i in range(len(data)):
            ques, label = data[i], data_labels[i]
            label_sample = list(set(random.sample(labels, int(len(labels)*0.8)) + label))
            prompt = instructions % (', '.join(label_sample).lower())
            input_text = f'Question: {ques}\nKnowledge points and skills: '
            output_text = ', '.join(label).lower()
            instruction_data.append({"instruction": prompt, "input": input_text, "output": output_text})
        sft_data += random.sample(instruction_data, rate[id])
    with open(f"sft_data_math.json", "w") as f:
        json.dump(instruction_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    datanames = ["codeforces", "leetcode", "newcoder"]
    sft_data_generate(datanames)
