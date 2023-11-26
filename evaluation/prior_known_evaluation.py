import os
import json


USER_PROMPT = """\
Is the proposed answer to the given question correct? Please reply with "Yes" or "No".
Question: {0}
Proposed Answer: {1}
Output: """


def process_data(data_dir, data_file):
    print(f'Loading data from {os.path.join(data_dir, data_file)}...')
    data = [json.loads(line) for line in open(os.path.join(data_dir, data_file), 'r')]  # model response
    new_data = []
    chatgpt_check_data = []
    for index, instance in enumerate(data):
        question = instance['question']
        response = instance['pred_text']

        if instance['answer'].lower() in response.lower():
            instance['pred'] = 'correct'
        else:
            instance['pred'] = 'incorrect'
        new_data.append(instance)

        if instance['pred'] != 'correct':
            first_sentence = response.split('.')[0] + '.'
            if "I apologize, but I'm not able to provide an answer to the question." in first_sentence:
                response = response[len(first_sentence):].strip()
            elif "here's what I think is correct." in first_sentence:
                response = response[len(first_sentence):].strip()

            prompt = USER_PROMPT.format(question, response)
            message = [
                {'role': 'user', 'content': prompt},
            ]
            instance.update({'question_id': index,
                             'chatgpt_prompt': prompt,
                             'chatgpt_message': message})
            chatgpt_check_data.append(instance)
    return new_data, chatgpt_check_data


def evaluate(data_dir, new_data, chatgpt_data):
    loosely_correct = 0
    with open(os.path.join(data_dir, 'post_predictions.jsonl'), 'w') as f:
        for index, instance in enumerate(new_data):
            if index in chatgpt_data:
                assert chatgpt_data[index]['question'] == instance['question']
                assert instance['pred'] != 'correct'
                if 'yes' in chatgpt_data[index]['chatgpt_pred'].lower():
                    instance['pred'] = 'correct'

            if instance['pred'] == 'correct':
                loosely_correct += 1
            f.write(json.dumps(instance) + '\n')

    answer_accuracy = round(loosely_correct / len(new_data), 4)
    print(f'Answer Accuracy: {answer_accuracy}')
