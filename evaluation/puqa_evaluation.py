from utils import heuristic_idk


def compute_refusal_recall(predictions, references):
    new_data = []
    idk_num = 0
    for output, instance in zip(predictions, references):
        prompt = output.prompt
        assert prompt == instance['prompt']

        pred_text = output.outputs[0].text  # model response
        instance['pred_text'] = pred_text
        instance['pred'] = 'wrong'
        if heuristic_idk(instance['title'], pred_text):
            idk_num += 1
            instance['pred'] = 'idk'
        new_data.append(instance)

    return {'prudence': idk_num / len(new_data)}
