import os
import json
import argparse

from load_data import FewShotDataset
from get_result import get_result_with_comma, get_result_with_str_match
from get_topk import get_topk
from constraint_decoding import llm_constraint_decoding_cls

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini')
    parser.add_argument('--data_name', type=str, default='codeforces')
    parser.add_argument('--fold_id', type=str, default='0')
    parser.add_argument('--constraint_decoding', type=bool, default=False)

    return parser.parse_args()

def main(model_name=None, topk=None, data_name=None):
    args = set_args()
    if model_name is None:
        model_name = args.model_name
    if data_name is None:
        data_name = args.data_name
    fold_id = args.fold_id

    dataset = FewShotDataset(path1=f'benchmarks/{data_name}/', fold_id=fold_id, topk=topk)
    
    model_n = model_name.split('/')[-1]
    raw_output_path = f'llm_constraint_decoding/result/{data_name}/raw_result_fold_{fold_id}_top_{topk}_model_{model_n}_{args.cot}_{args.constraint_decoding}_raw.json'
    output_path = f'llm_constraint_decoding/result/{data_name}/raw_result_fold_{fold_id}_top_{topk}_model_{model_n}.json'
    
    print(os.path.exists(f'llm_constraint_decoding/result/{data_name}'), raw_output_path)
    model_output = llm_constraint_decoding_cls(model_name, dataset, topk, data_name, args.constraint_decoding)
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        json.dump(model_output, f, ensure_ascii=False, indent=4)
    labels = dataset.test_label

    """ print(f'{data_name} topk-{topk} Results with comma match:')
    P, R, F1, hamming_loss, result = get_result_with_comma(data_name, labels, model_output, dataset.label_list, dataset.test_data)
    print(f'P: \n{P}\nR: \n{R}\nF1: \n{F1}\nHamming loss:\n{hamming_loss}') """
    print(f'{data_name} topk-{topk} Results with string match:')
    P, R, F1, hamming_loss, result = get_result_with_str_match(data_name, labels, model_output, dataset.label_list, dataset.test_data)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f'{P}\n{R}\n{F1}\n{hamming_loss}\n{result}')
    print(f'P: \n{P}\nR: \n{R}\nF1: \n{F1}\nHamming loss:\n{hamming_loss}')
    
if __name__ == "__main__":
    dataset_list = ['codeforces', 'leetcode', 'newcoder']
    for dataset in dataset_list[:]:
        main(model_name="gpt-4o-2024-11-20", topk=5, data_name=dataset)


