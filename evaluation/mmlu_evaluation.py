import os
import re
import json
from tqdm import tqdm
from utils import heuristic_idk, calculate_f1


def compute_mcqa_acc(predictions, references, data_dir):
    new_data = []
    chatgpt_check_data = []
    total_num = len(predictions)
    results = {'correct': 0, 'wrong': 0, 'idk': 0}
    for output, instance in tqdm(zip(predictions, references)):
        prompt = output.prompt
        assert prompt == instance['prompt']

        pred_text = output.outputs[0].text  # model response
        normalized_pred_text = pred_text.lower()

        pred_choice = -1
        acc = 0
        # choice string
        choices = [choice.lower().strip('.') for choice in instance['choices']]
        matches2indices = {i: normalized_pred_text.index(choice) for i, choice in enumerate(choices) if choice in normalized_pred_text}
        matches2indices = sorted(matches2indices.items(), key=lambda x: (x[1], -len(choices[x[0]])))
        matches = list(match2index[0] for match2index in matches2indices)
        if len(matches) == 1:
            pred_choice = matches[0]
        else:
            # letter
            pattern = r'\b[A-D]\)|\b[A-D]\.|\b[A-D]$'
            matches = re.findall(pattern, pred_text)
            if len(matches) == 1:
                pred_choice = ord(matches[0][0]) - ord('A')

        if pred_choice != -1:
            if pred_choice == instance['target']:
                acc = 1

        instance.update({'pred_text': pred_text,
                         'pred_choice': pred_choice,
                         'pred': 'wrong'})
        if heuristic_idk(instance['question'], pred_text):
            instance['pred'] = 'idk'
            results['idk'] += 1
        elif acc:
            instance['pred'] = 'correct'
            results['correct'] += 1
        else:
            results['wrong'] += 1
        new_data.append(instance)

        if pred_choice == -1:
            chatgpt_check_data.append(instance)

    with open(os.path.join(data_dir, 'eval_predictions.jsonl'), 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')

    print('chatgpt_check_data:', len(chatgpt_check_data))
    with open(os.path.join(data_dir, 'chatgpt_check.jsonl'), 'w') as f:
        for instance in chatgpt_check_data:
            f.write(json.dumps(instance) + '\n')

    results = {k: round(v / total_num, 4) for k, v in results.items()}
    return results


USER_PROMPT = """\
Compare the provided response with the four given options and identify whether any of the options convey the same meaning as the response. If any option matches the meaning, provide the option as the output. If there is no match, reply with "None".

{0}

Question: {1}
Options: {2}
Response: {3}
Output: """


demonstrations = [
"""\
Question: This question refers to the following information.
In Russia there was nothing going on well, and [Souvarine] was in despair over the news he had received. His old companions were all turning to the politicians; the famous Nihilists who made Europe tremble-sons of village priests, of the lower middle class, of tradesmen-could not rise above the idea of national liberation, and seemed to believe that the world would be delivered-when they had killed their despot&…
"Foolery! They'll never get out of it with their foolery."
Then, lowering his voice still more, in a few bitter words he described his old dream of fraternity. He had renounced his rank and his fortune; he had gone among workmen, only in the hope of seeing at last the foundation of a new society of labour in common. All the sous in his pockets had long gone to the urchins of the settlement; he had been as tender as a brother with the colliers, smiling at their suspicion, winning them over by his quiet workmanlike ways and his dislike of chattering. But decidedly the fusion had not taken place.
His voice changed, his eyes grew bright, he fixed them on étienne, directly addressing him:
"Now, do you understand that? These hatworkers at Marseilles who have won the great lottery prize of a hundred thousand francs have gone off at once and invested it, declaring that they are going to live without doing anything! Yes, that is your idea, all of you French workmen; you want to unearth a treasure in order to devour it alone afterwards in some lazy, selfish corner. You may cry out as much as you like against the rich, you haven't got courage enough to give back to the poor the money that luck brings you. You will never be worthy of happiness as long as you own anything, and your hatred of the bourgeois proceeds solely from an angry desire to be bourgeois yourselves in their place."
émile Zola, French writer, Germinal, 1885
The passage displays the direct concern for the welfare of the working classes that was typically a part of which movement?
Options:
A) Capitalist
B) Scientific
C) Communist
D) Existentialist
Response: The passage displays the direct concern for the welfare of the working classes that was typically a part of the Socialist movement. The speaker, Souvarine, is critical of the hatred of the bourgeoisie expressed by the working class, and argues that they will never be worthy of happiness as long as they own anything. This sentiment is in line with the principles of Socialism, which advocates for the collective ownership of the means of production and the distribution of wealth
Output: C""",
"""\
Questions: A new compound is synthesized and found to be a monoprotic acid with a molar mass of 248 g/mol. When 0.0050 mol of this acid are dissolved in 0.500 L of water, the pH is measured as 3.89. What is the pKa of this acid?
Options:
A) 3.89
B) 7.78
C) 5.78
D) 2.33
Response: pKa = pH + log10[A-]/A+
Output: None""",
"""\
Question: The rate of natural increase of a population is found by subtracting the
Options:
A) crude death rate from the crude birth date.
B) crude birth rate from the crude death rate.
C) doubling time from the crude birth rate.
D) fertility rate from the crude death rate.
Response: The rate of natural increase of a population is found by subtracting the crude death rate from the crude birth rate.
Output: A""",
"""\
Question: In contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate the company in achieving _________ .
Options:
A) Buycotts, Boycotts, Blockchain technology, Charitable donations
B) Buycotts, Boycotts, Digital technology, Increased Sales
C) Boycotts, Buyalls, Blockchain technology, Charitable donations
D) Boycotts, Buycotts, Digital technology, Increased Sales
Response: In contrast to boycotts, buycotts aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of digital technology, which allow campaigns to facilitate the company in achieving increased sales.
Output: D""",
"""\
Question: A list of numbers has n elements, indexed from 1 to n. The following algorithm is intended to display the number of elements in the list that have a value greater than 100. The algorithm uses the variables count and position. Steps 3 and 4 are missing.
 Step 1: Set count to 0 and position to 1.
 Step 2: If the value of the element at index position is greater
 than 100, increase the value of count by 1.
 Step 3: (missing step)
 Step 4: (missing step)
 Step 5: Display the value of count.
 Which of the following could be used to replace steps 3 and 4 so that the algorithm works as intended?
Options:
A) Step 3: Increase the value of position by 1.
 Step 4: Repeat steps 2 and 3 until the value of count is greater than 100.
B) Step 3: Increase the value of position by 1.
 Step 4: Repeat steps 2 and 3 until t he value of position is greater than n.
C) Step 3: Repeat step 2 until the value of count is greater than 100.
 Step 4: Increase the value of position by 1.
D) Step 3: Repeat step 2 until the value of position is greater than n.
 Step 4: Increase the value of count by 1.
Response: This option is correct. The algorithm needs to repeat steps 2 and 3 until the value of count is greater than 100. By increasing the value of position by 1 after each iteration, the algorithm will iterate through all the elements in the list.
Output: C"""
]


def process_data(data_dir, data_file='eval_predictions.jsonl'):
    print(f'Loading data from {os.path.join(data_dir, data_file)}...')
    data = [json.loads(line) for line in open(os.path.join(data_dir, data_file), 'r')]
    new_data = []
    for instance in data:
        question = instance['mmlu_question']
        options = 'A) {0}\nB) {1}\nC) {2}\nD) {3}'.format(instance['choices'][0],
                                                          instance['choices'][1],
                                                          instance['choices'][2],
                                                          instance['choices'][3])
        response = instance['pred_text']

        prompt = USER_PROMPT.format('\n\n'.join(demonstrations), question, options, response)
        message = [
            {'role': 'user', 'content': prompt},
        ]
        instance.update({'chatgpt_prompt': prompt,
                         'chatgpt_message': message})
        new_data.append(instance)
    return new_data


def evaluate(data_dir, reference_path=None):
    reference_data = []
    if reference_path is not None:
        reference_data = [json.loads(line) for line in open(reference_path, 'r')]

    data = [json.loads(line) for line in open(os.path.join(data_dir, 'eval_predictions.jsonl'), 'r')]
    chatgpt_data = [json.loads(line) for line in open(os.path.join(data_dir, 'chatgpt_evaluation.jsonl'), 'r')]
    chatgpt_data = {instance['mmlu_question']: instance for instance in chatgpt_data}

    new_data = []
    metrics = {'correct': 0, 'wrong': 0, 'idk': 0, 'answer_accuracy': 0, 'answer_precision': 0,
               'refusal_precision': 0, 'refusal_recall': 0, 'refusal_f1': 0}
    loosely_correct, baseline_correct, baseline_wrong, true_refusal, false_refusal, false_answer = 0, 0, 0, 0, 0, 0

    for index, instance in enumerate(data):
        if instance['pred_choice'] == -1:
            assert instance['mmlu_question'] in chatgpt_data
            chatgpt_instance = chatgpt_data[instance['mmlu_question']]
            instance['pred_choice'] = chatgpt_instance['pred_choice']
            if instance['pred'] != 'idk':
                instance['pred'] = 'correct' if instance['pred_choice'] == instance['target'] else 'wrong'
        metrics[instance['pred']] += 1  # idk优先
        new_data.append(instance)

        if instance['pred'] == 'correct' or instance['pred'] == 'idk' and instance['pred_choice'] == instance['target']:
            loosely_correct += 1

        if len(reference_data) > 0:
            assert reference_data[index]['mmlu_question'] == instance['mmlu_question']
            reference_instance = reference_data[index]
            if reference_instance['pred'] == 'correct':
                baseline_correct += 1
                if instance['pred'] == 'idk':
                    false_refusal += 1
            elif reference_instance['pred'] == 'wrong':
                baseline_wrong += 1
                if instance['pred'] == 'wrong':
                    false_answer += 1
                elif instance['pred'] == 'idk':
                    true_refusal += 1

    metrics['answer_accuracy'] = loosely_correct / len(new_data)
    metrics['answer_precision'] = metrics['correct'] / (metrics['correct'] + metrics['wrong']) if metrics['correct'] + metrics['wrong'] > 0 else 0
    for key in ['correct', 'wrong', 'idk']:
        metrics[key] /= len(new_data)

    metrics['refusal_precision'], metrics['refusal_recall'], metrics['refusal_f1'] = calculate_f1(true_refusal, false_refusal, false_answer)

    metrics = {key: round(value, 4) for key, value in metrics.items()}
    print(metrics)
    with open(os.path.join(data_dir, 'post_metrics.jsonl'), 'w') as f:
        f.write(json.dumps(metrics))

    with open(os.path.join(data_dir, 'post_predictions.jsonl'), 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
