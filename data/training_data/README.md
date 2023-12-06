# Training Data Overview

In addition to the provided training datasets, you have the flexibility to construct additional training data for various models using specific datasets. This can be achieved by exloiting the codes provided in [``evaluation``]().

## Key Data Fields
The important fileds of the training dataset are as follows.
* ``greedy_pred_text``: The unaligned model's response at temperature = 0 under typical question answering prompt.
* ``greedy_label``: If the greedy_pred_text is correct, then "known", else "unknown". 
* ``sampling_pred_text``: The unaligned model's 10 sampled responses at temperature = 1 under typical question answering prompt. 
* ``sampling_labels``: Corresponding labels to sampling_pred_text. 
* ``sampling_knowns``: The number of "knowns" in sampling_labels.