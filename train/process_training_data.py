import os
import copy
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from enum import Enum


class SFTMethod(Enum):
    SFT_BASELINE = "sft-baseline"
    ABSOLUTE = "absolute"
    CONFIDENCE_NUM = "confidence-num"
    CONFIDENCE_VERB = "confidence-verb"
    MULTISAMPLE = "multisample"


prompts = [
{'prompt':  # typical QA
"""\
Q: {0}
A: """,
'completion': "I apologize, but I'm not able to provide an answer to the question."},
{'prompt':  # honest QA
"""\
Answer the question. If you don't know the answer to the question, it is appropriate to say "I apologize, but I'm not able to provide an answer to the question."
Q: {0}
A: """,
'completion': "I apologize, but I'm not able to provide an answer to the question."},
{'prompt':  # confidence-num
"""\
Answer the question. If you don't know the answer to the question, it is appropriate to say "I apologize, but I'm not able to provide an answer to the question."
Q: {0}
A: """,
'completion': [
"I apologize, but I'm not able to provide an answer to the question.",
"I'm only about {}% confident to answer the question correctly, but based on my understanding and knowledge, here's what I think is correct.",
"I'm about {}% confident to answer the question correctly, and based on my understanding and knowledge, here's what I think is correct."]},
{'prompt':  # confidence-verb
"""\
Answer the question. If you don't know the answer to the question, it is appropriate to say "I apologize, but I'm not able to provide an answer to the question."
Q: {0}
A: """,
'completion': [
"I apologize, but I'm not able to provide an answer to the question.",
"I'm really not sure about this, but ",
"I'm not completely sure about this, but ",
"I don't have strong feelings either way, but ",
"I'm fairly confident that ",
"I'm absolutely certain that ",]},
]


@dataclass
class DataArguments:
    data_dir: str = field(default='data', metadata={"help": "The directory to save processed data."})
    dataset_name: str = field(default='triviaqa')
    sft_method: SFTMethod = field(default=SFTMethod.ABSOLUTE)

    train_data_path: str = field(default=None)
    eval_data_path: str = field(default=None)
    prompt_id: int = field(default=0)
    data_max_length: int = field(default=1024)
    refresh: bool = field(default=False, metadata={"help": "Whether to refresh the data."})


class MyDataset(Dataset):
    def __init__(self, data_args, split):
        super().__init__()
        self.data_args = data_args
        self.split = split
        self.sft_method = data_args.sft_method
        self.prompt_id = data_args.prompt_id

        data_tag = data_args.sft_method
        if data_args.prompt_id != 0:
            data_tag += f'_p{data_args.prompt_id}'

        save_dir = os.path.join(data_args.data_dir, data_args.dataset_name, data_tag)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'{split}.pt')
        if data_args.refresh or not os.path.exists(save_file):
            if split == 'train':
                path = data_args.train_data_path
            else:
                path = data_args.eval_data_path
            dataset = [json.loads(line) for line in open(path, 'r')]

            if data_args.sft_method == SFTMethod.CONFIDENCE_NUM or data_args.sft_method == SFTMethod.CONFIDENCE_VERB:
                self.data = self.confidence_process(dataset, save_file)
            elif data_args.sft_method == SFTMethod.MULTISAMPLE:
                self.data = self.multisample_process(dataset, save_file)
            else:
                self.data = self.process(dataset, save_file)
        else:
            print('Loading data from', save_file)
            self.data = torch.load(save_file)
        # self.data = self.data[:10]  # For debug
        print('Data size:', len(self.data))
        print('Data format:', self.data[0])
        print('Data format:', self.data[-1])

    def process(self, dataset, save_file):
        data = []
        print_flag = True
        for instance in tqdm(dataset):
            new_instance = copy.deepcopy(instance)
            source = prompts[self.prompt_id]['prompt'].format(instance['question'])

            if self.sft_method == SFTMethod.SFT_BASELINE:
                if instance['greedy_label'] == 'known':
                    output = instance['greedy_pred_text']
                else:
                    if 'gold_answer' in instance:
                        output = instance['gold_answer'].capitalize() + '.'
                    elif 'target' in instance:
                        output = f"{chr(instance['target']) + 'A'}) {instance['choices'][instance['target']]}"
                        if print_flag:
                            print(output)
                            print_flag = False
                    else:
                        raise NotImplementedError
                    for sampling_pred_text, sampling_label in zip(instance['sampling_pred_text'], instance['sampling_labels']):
                        if sampling_label == 'known':
                            output = sampling_pred_text
                            break
            elif self.sft_method == SFTMethod.ABSOLUTE:
                if instance['greedy_label'] == 'known':
                    output = instance['greedy_pred_text']
                else:
                    output = prompts[self.prompt_id]['completion']
                    for sampling_pred_text, sampling_label in zip(instance['sampling_pred_text'], instance['sampling_labels']):
                        if sampling_label == 'known':
                            output = sampling_pred_text
                            break
            else:
                raise NotImplementedError

            new_instance['input'] = source
            new_instance['output'] = output
            data.append(new_instance)

        torch.save(data, save_file)
        print('Saving data to', save_file)
        return data

    def multisample_process(self, dataset, save_file):
        data = []
        for instance in tqdm(dataset):
            assert len(instance['sampling_pred_text']) == len(instance['sampling_labels']) == 10
            for sampling_pred_text, sampling_label in zip(instance['sampling_pred_text'], instance['sampling_labels']):
                new_instance = copy.deepcopy(instance)
                source = prompts[self.prompt_id]['prompt'].format(instance['question'])
                if sampling_label == 'known':
                    output = sampling_pred_text
                else:
                    output = prompts[self.prompt_id]['completion']

                new_instance['input'] = source
                new_instance['output'] = output
                data.append(new_instance)

        torch.save(data, save_file)
        print('Saving data to', save_file)
        return data

    def confidence_process(self, dataset, save_file):
        print_flag = [True, True, True, True, True, True]
        data = []
        for instance in tqdm(dataset):
            new_instance = copy.deepcopy(instance)
            source = prompts[self.prompt_id]['prompt'].format(instance['question'])

            response = instance['greedy_pred_text']
            if instance['greedy_label'] != 'known':
                for sampling_pred_text, sampling_label in zip(instance['sampling_pred_text'], instance['sampling_labels']):
                    if sampling_label == 'known':
                        response = sampling_pred_text
                        break

            if instance['sampling_knowns'] == 0:
                output = prompts[self.prompt_id]['completion'][0]
                if print_flag[0]:
                    print(output)
                    print_flag[0] = False
            elif self.sft_method == SFTMethod.CONFIDENCE_NUM:
                if instance['sampling_knowns'] <= 5:
                    confidence = prompts[self.prompt_id]['completion'][1].format(instance['sampling_knowns'] * 10)
                    output = confidence + ' ' + response
                    if print_flag[1]:
                        print(output)
                        print_flag[1] = False
                else:
                    confidence = prompts[self.prompt_id]['completion'][2].format(instance['sampling_knowns'] * 10)
                    output = confidence + ' ' + response
                    if print_flag[2]:
                        print(output)
                        print_flag[2] = False
            elif self.sft_method == SFTMethod.CONFIDENCE_VERB:
                if not (response in ['A', 'B', 'C', 'D'] or response[0] in ['A', 'B', 'C', 'D'] and response[1] in [')', '.']):
                    response = response[0].lower() + response[1:]
                if instance['sampling_knowns'] <= 2:
                    output = prompts[self.prompt_id]['completion'][1] + response
                    if print_flag[1]:
                        print(output)
                        print_flag[1] = False
                elif instance['sampling_knowns'] <= 4:
                    output = prompts[self.prompt_id]['completion'][2] + response
                    if print_flag[2]:
                        print(output)
                        print_flag[2] = False
                elif instance['sampling_knowns'] <= 6:
                    output = prompts[self.prompt_id]['completion'][3] + response
                    if print_flag[3]:
                        print(output)
                        print_flag[3] = False
                elif instance['sampling_knowns'] <= 8:
                    output = prompts[self.prompt_id]['completion'][4] + response
                    if print_flag[4]:
                        print(output)
                        print_flag[4] = False
                else:
                    output = prompts[self.prompt_id]['completion'][5] + response
                    if print_flag[5]:
                        print(output)
                        print_flag[5] = False
            else:
                raise NotImplementedError

            new_instance['input'] = source
            new_instance['output'] = output
            data.append(new_instance)

        torch.save(data, save_file)
        print('Saving data to', save_file)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    parser = HfArgumentParser((DataArguments))
    data_args = parser.parse_args_into_dataclasses()
    data_args.prompt_id = 1
    data_args.train_data_path = ''

    train_dataset = MyDataset(data_args, split='train')
