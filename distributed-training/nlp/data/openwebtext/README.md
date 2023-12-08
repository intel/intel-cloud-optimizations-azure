## openwebtext dataset

Upon successful completion of the [`prepare.py`](prepare.py) script, two files will be generated:

- `train.bin`: This file will be approximately 17GB (~9B tokens) in size.
- `val.bin`: This file will be around 8.5MB (~4M tokens) in size.

This came from 8,013,769 documents in total.

References:

- OpenAI's WebText dataset is discussed in [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset