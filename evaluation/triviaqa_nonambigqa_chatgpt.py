import os
import json


# 1. Extract short answers from model's responses.
USER_PROMPT1 = """\
Given a question and a piece of text, if the text does not contain an answer to the question, output "no answer"; otherwise, extract the answer from the text.

{0}

Question: {1}
Text: {2}
Output: """

# Note: triviaqa uses 5-shot (the first 5 demonstrations), and non-ambigqa uses 6-shot.
demonstrations1 = [
"""\
Question: What was the last US state to reintroduce alcohol after prohibition?
Text: The last US state to reintroduce alcohol after prohibition was Mississippi. Mississippi legalized alcohol on August 17, 1933, making it the last state to do so. Prohibition had been enacted nationwide through the 18th Amendment to the US Constitution in 1919, but it was repealed in 1933 with the passage of the 21st Amendment, which allowed individual
Output: Mississippi""",
"""\
Question: What was the name of Michael Jackson's autobiography written in 1988?
Text: Michael Jackson did not write an autobiography in 1988.
Output: no answer""",
"""\
Question: Who was the director of the CIA from 1976-81?
Text: Stansfield Turner.
Output: Stansfield Turner""",
"""\
Question: Which 90s sci fi series with James Belushi was based on Bruce Wagner's comic strip of the same name?
Text: The 90s sci-fi series based on Bruce Wagner's comic strip was "The Adventures of Buckaroo Banzai Across the 8th Dimension." It starred James Belushi as the titular character and aired in 1984.
Output: The Adventures of Buckaroo Banzai Across the 8th Dimension.""",
"""\
Question: Art Garfunkel trained for which profession although he didn't qualify?
Text: Art Garfunkel trained to be a pharmacist, but he did not qualify.
Output: pharmacist""",
"""\
Question: Name the landforms that form the boundaries of the peninsular plateau?
Text: The landforms that form the boundaries of the Peninsular Plateau are: 1. The Western Ghats: a range of hills that runs along the western edge of the peninsula, from the Satpura Range in the north to Kanyakumari in the south. 2. The Eastern Ghats: a range of hills that runs along the eastern edge of the peninsula, from the Satpura Range in the north to the Nilgiri Hills
Output: The Western Ghats, The eastern Ghats"""
]

def process_data1(data_dir, data_file):
    print(f'Loading data from {os.path.join(data_dir, data_file)}...')
    data = [json.loads(line) for line in open(os.path.join(data_dir, data_file), 'r')]
    new_data = []
    for instance in data:
        if 'triviaqa' in data_dir.lower():
            question = instance['question'].strip('"')
            answer = instance['pred_text'].strip('"').split('\n')[0].strip()
            demo = '\n\n'.join(demonstrations1[:5])
        elif 'nonambigqa' in data_dir.lower():
            question = instance['question']
            answer = instance['pred_text'].replace('\n\n', ' ').replace('\n', ' ').strip()
            demo = '\n\n'.join(demonstrations1)
        else:
            raise NotImplementedError

        prompt = USER_PROMPT1.format(demo, question, answer)
        message = [
            {'role': 'user', 'content': prompt},
        ]
        instance.update({'chatgpt_prompt': prompt,
                         'chatgpt_message': message})
        new_data.append(instance)
    return new_data


# 2. Compare extracted short answers and gold answers.
USER_PROMPT2 = """\
Please rate the consistency between the reference answer and the proposed answer on a scale of 0 to 1. A rating of 0 indicates inconsistency, while a rating of 1 indicates perfect consistency.

{0}

Question: {1}
Reference Answer: {2}
Proposed Answer: {3}
Score: """


demonstrations2 = [
"""\
Question: In which country is the Sky Train Rail bridge?
Reference Answer: Canada
Proposed Answer: Thailand
Score: 0""",
"""\
Question: What color is the lowest level of the Homeland Security Advisory System?
Reference Answer: Green
Proposed Answer: Blue
Score: 0""",
"""\
# Question: After the United States and the Soviet Union, what country became the third in the world to test an atom bomb (in 1952)?
# Reference Answer: Great Britain
# Proposed Answer: United Kingdom
# Score: 1""",
"""\
Question: What was Eddie Murphy's first movie?
Reference Answer: 48 Hours
Proposed Answer: 48 Hrs.
Score: 1""",
"""\
Question: How long do NFL football teams have to get a play off (the play clock)?
Reference Answer: 40 seconds
Proposed Answer: 30 seconds
Score: 0""",]


def process_data2(data_dir, data_file):
    print(f'Loading data from {os.path.join(data_dir, data_file)}...')
    data = [json.loads(line) for line in open(os.path.join(data_dir, data_file), 'r')]
    new_data = []
    for instance in data:
        question = instance['question']
        answer = instance['chatgpt_pred'].strip()

        prompt = USER_PROMPT2.format('\n\n'.join(demonstrations2), question, instance['gold_answer'], answer)
        message = [
            {'role': 'user', 'content': prompt},
        ]
        instance.update({'chatgpt_prompt': prompt,
                         'chatgpt_message': message})
        new_data.append(instance)
    return new_data
