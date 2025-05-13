from torch.utils.data import Dataset
import json

def getDef(def_file):
    defs={}
    f=open(def_file, 'r', encoding='utf-8')
    text=f.readlines()
    f.close()
    for i in text:
        temp=i.strip().split('\t')
        if len(temp) == 2:
            defs[int(temp[0])] = temp[1]
        else:
            print(temp[0])
            defs[int(temp[0])] = ''
    return defs

class FewShotDataset(Dataset):
    def __init__(self, path1, fold_id='0', topk=3, test_path=None):
        self.topk = topk
        self.id2ques = getDef(path1 + 'ques.tsv')
        self.id2label = getDef(path1 + 'tag.tsv')
        self.label_list = list(self.id2label.values())
        self.data = []
        self.label = []
        #读取完整的数据集，用于计算相似度以及获取top-k数据
        #f = open(f'{path1}data.tsv', 'r', encoding='utf-8')
        f = open(f'{path1}/fold_0/train.tsv', 'r', encoding='utf-8')
        text = f.readlines()
        f.close()
        for i in text:
            line = i.strip().split('\t')
            self.data.append(self.id2ques[int(line[0])])
            temp = []
            labels = len(line) - 1
            for j in range(labels):
                if line[j + 1] == '1':
                    temp.append(self.id2label[int(j)])
            self.label.append(temp)
        #读取测试数据
        self.test_data = []
        self.test_label = []
        self.test_ids = []
        if not test_path:
            f = open(f'{path1}fold_{fold_id}/test.tsv', 'r', encoding='utf-8')
            text = f.readlines()
            f.close()
        else:
            f = open(test_path, 'r', encoding='utf-8')
            text = f.readlines()
            f.close()
        
        for i in text:
            line = i.strip().split('\t')
            self.test_ids.append(int(line[0]))
            self.test_data.append(self.id2ques[int(line[0])])
            temp = []
            labels = len(line) - 1
            for j in range(labels):
                if line[j + 1] == '1':
                    temp.append(self.id2label[int(j)])
            self.test_label.append(temp)
        
        self.topk_data = [[] for i in range(len(self.test_data))]
        
    def load_topk_data(self, path1, cot=False):
        # 加载相似度最高的 top-k 数据
        self.ques_sims = json.load(open(path1 + 'ques_similarity.json', 'r', encoding='utf-8'))
        if cot:
            topk_result = []
            with open(f'{path1}/generated_examples.json', 'r', encoding='utf-8') as f:
                for i in range(len(self.ques_sims)):
                    line = json.loads(f.readline())
                    assert line[0]['index'] == i
                    topk_result.append(line[-1]['content'])
        #print(topk_result[0])
        self.topk_data = []
        for text_id in self.test_ids:
            temp = []
            cunt = 0
            # 对于每行测试数据，获取其相似度最高的非测试集top-k 条数据
            for line in self.ques_sims[str(text_id)]:
                if cunt >= self.topk:
                    break
                if line[0] in self.test_ids:
                    continue
                ids = int(line[0])
                if cot:
                    temp.append((self.data[ids], topk_result[ids]))
                else:
                    temp.append((self.data[ids], ', '.join(self.label[ids])))
                cunt += 1
            self.topk_data.append(temp)
        
            
    def __getitem__(self, idx):
        return self.test_data[idx], self.test_label[idx]
    
    def __len__(self):
        return len(self.test_data)