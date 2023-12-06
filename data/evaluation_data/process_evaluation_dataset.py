import os
import json
from datasets import load_dataset


def construct_nonambigqa():
    new_data = []

    for split in ['train', 'validation']:
        data = load_dataset('ambig_qa', name='light', split=split)
        for instance in data:
            if len(instance['annotations']['type']) > 1 or instance['annotations']['type'][0] != 'singleAnswer':
                continue

            new_instance = {}
            new_instance['question_id'] = instance['id']
            new_instance['question'] = instance['question']
            answers = instance['annotations']['answer']
            assert len(answers) == 1
            new_instance['answers'] = answers[0]
            new_instance['gold_answer'] = new_instance['answers'][0]

            new_data.append(new_instance)

    print(len(new_data))
    with open('nonambigqa.jsonl', 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')


def construct_mmlu(split):
    MCQA_PROMPT = '{0}\nA) {1}\nB) {2}\nC) {3}\nD) {4}'

    new_data = []
    mmlu_split = 'dev' if split == 'train' else 'test'
    data = load_dataset('cais/mmlu', name='all', split=mmlu_split)
    for instance in data:
        new_instance = {'mmlu_question': instance['question'],
                        'subject': instance['subject'],
                        'choices': instance['choices'],
                        'target': instance['answer']}
        new_instance['question'] = MCQA_PROMPT.format(new_instance['mmlu_question'],
                                                                  new_instance['choices'][0],
                                                                  new_instance['choices'][1],
                                                                  new_instance['choices'][2],
                                                                  new_instance['choices'][3])
        new_data.append(new_instance)

    print(len(new_data))
    with open('mmlu.jsonl', 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
