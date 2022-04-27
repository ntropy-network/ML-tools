# Elyra customized transaction classification model pipeline
Notebooks and Elyra pipelines file accompanying the blog post

## Files
- `getdata.ipynb`: Fetches transaction train and test data from S3
- `train.ipynb`: Trains a customized transaction classification model on the train data
- `evaluate.ipynb`: Loads the trained model and evaluates it on the test data
- `ntropy-pipeline.pipeline`: Elyra pipeline tying the notebooks together