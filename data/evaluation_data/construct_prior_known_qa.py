import os
import json
import re
from rouge_score import rouge_scorer
from multiprocessing import Pool
from functools import partial
import numpy as np
from typing import List, Literal, TypedDict
from utils import random_sample
num_cpus=16


Role = Literal["system", "user", "assistant"]
class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
instruction = "Please generate 20 simple, knowledge-intensive question answering problems and their corresponding correct answers on the topic of \"{}\". Each problem should be in the format of \"Q: <question>\nA: <answer>\". The answers should be short phrases."


def generate_prompt():
    data = []
    for topic in ['Celebrities & Entertainment News',
                  'Comics & Animation',
                  'Movies',
                  'Music & Audio',
                  'Performing Arts',
                  'TV & Video',
                  'Visual Art & Design',
                  'Transportation',
                  'Beauty & Fitness',
                  'Books & Literature',
                  'Business & Industrial',
                  'Computers & Electronics',
                  'Finance',
                  'Food & Drink',
                  'Games',
                  'Health',
                  'History & News',
                  'People & Society',
                  'Animals',
                  'Science',
                  'Sports',
                  'Geography & Travel']:
        prompt = instruction.format(topic)

        dialog = [{'role': 'user', 'content': prompt}]
        prompt = f"{B_INST} {(dialog[0]['content']).strip()} {E_INST}"
        data.append({'prompt': prompt,
                     'topic': topic})
    return data


# deduplicate similar questions, filter quetions whose answer is longer than 5 tokens
def filter_questions(data_dir):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    known_questions = []
    all_questions = []
    all_question_tokens = []
    for seed in range(20):
        data = [json.loads(line) for line in open(os.path.join(data_dir, f'seed{seed}', 'eval_predictions.jsonl'), 'r')]
        for instance in data:
            pattern = f'Q: (.*?)\nA: (.*?)\.'
            matches = re.findall(pattern, instance['pred_text'])
            for match in matches:
                known_question = {'topic': instance['topic'], 'question': match[0], 'answer': match[1]}
                if len(known_question['answer'].split()) > 5:
                    continue
                if len(all_question_tokens) > 0:
                    new_question_tokens = scorer._tokenizer.tokenize(known_question['question'])
                    with Pool(num_cpus) as p:
                        rouge_scores = p.map(
                            partial(rouge_scorer._score_lcs, new_question_tokens),
                            all_question_tokens,
                        )
                    rouge_scores = [score.fmeasure for score in rouge_scores]
                    most_similar_instructions = {
                        all_questions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                    }
                    if max(rouge_scores) > 0.7:
                        continue
                    known_question['most_similar_questions'] = most_similar_instructions
                    known_question['avg_similarity_score'] = float(np.mean(rouge_scores))
                else:
                    new_question_tokens = scorer._tokenizer.tokenize(known_question['question'])
                    known_question['most_similar_questions'] = {}
                    known_question['avg_similarity_score'] = -1

                all_questions.append(known_question['question'])
                all_question_tokens.append(new_question_tokens)
                known_questions.append(known_question)
        print(f'seed{seed} done')

    print(f'# of known questions: {len(known_questions)}')
    with open(os.path.join(data_dir, 'unverified_known_questions.jsonl'), 'w') as f:
        for instance in known_questions:
            f.write(json.dumps(instance) + '\n')


if __name__ == '__main__':
    """
    1. generate model's response to the prompt by generate_prompt()
    2. filter_questions()
    3. use evaluation/prior_known_evaluation.py to verify unfiltered_known_questions.jsonl
    4. save to prior_known_dataset_{}.jsonl; note that the dataset is different for different models
    """
    pass
