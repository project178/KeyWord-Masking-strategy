# KeyWord Masking strategy

This is the code for the paper "Could key-word masking strategy improve language model?". The paper investigates the effectiveness of a key-word masking strategy in improving language models. The code allows you to train a model on your own data and lexicons using this strategy. For more details, please read the paper.

## Usage

To train a model using the key-word masking strategy, you can run the `train.py` script with the following command-line arguments:


```python train.py --data=<path_to_training_data> --model_name=<name_of_pretrained_model> --model_path=<path_of_pretrained_model> --path_to_save=<path_to_save_trained_model> --entities=<entities_types_to_predict> --lexicons=<path_to_lexicons>```


`data`: path to your training data. The training data should be in a format that is compatible with the language model you are using (e.g., a text file with one sentence per line).

`model_name`: name of the pre-trained language model to use. You can choose from a range of pre-trained models provided by Hugging Face.

`model_name`: path of the pre-trained language model to use. You can continue the training from the checkpoint (`.pt` file).

`path_to_save`: path where the trained model will be saved.

`entities`: entities types to predict while testing the model.

`lexicons`: path to the lexicons you want to use for key-word masking. The lexicons should be in a text file format, with one keyword per line.

## Examples

Here are some examples of how to run the `train.py` script:

Train a model on the news-corpus dataset using the bert-base-multilingual-cased pre-trained model, lexicons `lexicon1.txt`and `lexicon3.txt` and save the trained model to ./models:

```python train.py --data=./corpora/news-corpus --model_name=bert-base-multilingual-cased --path_to_save=./models --lexicons=./lexicons/lexicon1.txt_./lexicons/lexicon3.txt```

## Requirements

This code requires the following Python packages:

- `transformers`

- `torch`

You can install these packages using `pip`.
