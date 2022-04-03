# Parallel cross encoders

 This project proposes a technique to speed up cross-encoders computation (both during training and inference).
It uses a special attention mechanism that allows the network to predict over several labels in parallel.

 This codebase uses the AG News text classification dataset from HuggingFace (4 labels).
You can find more information there: https://github.com/huggingface/datasets/tree/master/datasets/ag_news

## How to run the experiment ?

 Install project using poetry or pip:

    `poetry install` or `pip install .`

 You can use Google Colab for a free GPU. In that case, you should upload the content of this folder on your google Drive and run the notebooks on Colab.

 To train the models, run the `Train.ipynb` notebook

 To evaluate the models, run the `Test.ipynb` notebook

