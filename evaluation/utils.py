import re


def heuristic_idk(question, pred_text):
    idk_response = "I apologize, but I'm not able to provide an answer to the question".lower()
    if idk_response in pred_text.lower():
        return True

    question = question.lower()
    pred_text = pred_text.lower()
    if any(idk_expr in pred_text and idk_expr not in question for idk_expr in [
        'i apologize',
        'not aware of',
        'i\'m not familiar with',
        'i am not familiar with',
        'n\'t make sense',
        'not make sense',
        'i\'m not able to',
        'i am not able to',
        'however, i must point out']):
        return True
    return False


def correct_by_chatgpt_score(instance):
    # assert instance['label'] == 'unknown'
    try:
        chatgpt_score = float(instance['chatgpt_score'])
    except:
        pattern = r'(\d+[\.\d+]?)'
        match = re.search(pattern, instance['chatgpt_score'])
        if match:
            chatgpt_score = float(match.group(1))
        else:
            chatgpt_score = 0.
    return chatgpt_score > 0.7


def calculate_f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1
