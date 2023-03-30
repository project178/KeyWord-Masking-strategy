#standard
import os
import re
import datetime
import threading
from typing import List, Iterable, Union
os.chdir("/mnt/beegfs/projects/beyond/draft")

#ml
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoConfig, AdamW

#progress
import telebot
import git


class Preprocessor:
    """
    This class provides functionality to preprocess text for further treatment by a model.

    Attributes
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer used for preprocessing

    Methods
    -------
    split_by_sep(tokenized, sep, max_len=512)
        splits a tokenized text into smaller pieces by any separator from a list.
    cut(tokenized, max_len=512)
        makes the texts shorter by dividing them into approximately equal parts.
    mask_random(sentence, masks_proportion = 0.15)
       masks random entities in a tokenized sentence.
    preprocess(texts, labels, masks_proportion=0.15, max_len=512, sep=None)
       preprocesses the corpus for further usage by a model by tokenizing the texts,
       splitting them into smaller pieces if necessary, and masking random tokens.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def split_by_sep(
            self,
            tokenized: List[str],
            sep: Union[Iterable[str], str] = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
            max_len: int = 512,
    ) -> List[List[str]]:

        """
        Splits a tokenized text into smaller pieces by any separator from a list.

        :param tokenized: tokenized text
        :param sep: separators
        :param max_len: the maximum number of tokens in shortened texts
        :return: tokenized text split by the separators
        """

        old_len = len(tokenized)
        sep_num = old_len//max_len
        new_len = old_len//(sep_num + 1)
        all_indices = [i + 1 for i, token in enumerate(tokenized) if token in sep]
        selected_indices, j = [], 1
        for i, index in enumerate(all_indices):
            if index > new_len * j:
                if index - new_len < new_len - all_indices[i-1]: selected_indices.append(index)
                else: selected_indices.append(all_indices[i-1])
                j += 1
                if j > sep_num: break
        return [tokenized[start:end-1] for start, end in zip([0] + selected_indices, selected_indices + [old_len])]

    def _maximalize_length(self, text: List[List[str]], max_len: int) -> List[List[str]]:
        """
        Maximizes the length of text parts while ensuring they do not exceed the specified maximum length
        :param text: tokenized text split into small parts
        :type text: List[List[str]]
        :param max_len: the maximum number of tokens in one piece of a text
        :type max_len: int
        :return: "reshuffled" tokenized text
        """

        new_text = []
        cur_len = 0
        cur_part = []
        for part in text:
            length = len(part)
            cur_len += length
            if cur_len > max_len:
                new_text.append(cur_part)
                cur_len = length
                cur_part = part
            else:
                cur_part += part

        return new_text

    def _define_token_of_word_beginning(self, test_word: str = "aaaaaa", universal_token: str = "a"):
        """Identifies the symbol specific to the first token of a word according to the current tokenizer."""
        first, not_first = self.tokenizer.tokenize(test_word)[:2]
        first_start, not_first_start = first[:first.find(universal_token)], not_first[:not_first.find(universal_token)]
        if first_start:
            if first_start == -1 and not_first_start == 0:
                self._is_a_first_token_of_a_word = lambda token: token == first_start
            else:
                self._is_a_first_token_of_a_word = lambda token: token.startswith(first_start)
        elif not_first_start:
            self._is_a_first_token_of_a_word = lambda token: not (token.startswith(not_first_start))

    def _find_next_sep_pos(self, tokenized: List[str], cur_pos: int) -> int:
        """Finds next index to split a text by length"""
        for i, token in enumerate(tokenized[cur_pos:]):
            if self._is_a_first_token_of_a_word(token):
                return cur_pos + i
        return len(tokenized)

    def cut(self, tokenized: List[str], max_len: int = 512) -> List[List[str]]:
        """
        Divides input texts into roughly equal parts to make them shorter.
        :param tokenized: tokenized text
        :param max_len: the maximum number of tokens in shortened texts
        :return: shortened texts
        """
        old_len = len(tokenized)
        short_texts_num = old_len//max_len + 1
        new_length = old_len//short_texts_num
        short = []
        for i in range(short_texts_num):
            shorter = tokenized[new_length*i:self._find_next_sep_pos(tokenized, new_length*(i+1))]
            if 0 < len(shorter) <= max_len:
                short.append(shorter)
        del tokenized

        return short

    def mask_random(self, sentence: List[str], masks_proportion: float = 0.15) -> List[str]:
        """
        Masks random entities in a tokenized text by replacing a portion of the entities with a mask token.
        :param sentence: tokenized text
        :param masks_proportion: proportion of masked tokens
        :return: tokenized text with several tokens replaced by the mask
        """
        while self.tokenizer.mask_token not in sentence and masks_proportion > 0:
            masks = torch.rand(len(sentence)) < masks_proportion
            for mask_index, val in enumerate(masks):
                if val:
                    for i, token in enumerate(sentence[mask_index::-1]):
                        if self._is_a_first_token_of_a_word(token) or mask_index == 0:
                            start = mask_index - i
                            break
                    else:
                        continue
                    for i, token in enumerate(sentence[mask_index + 1:], 1):
                        if self._is_a_first_token_of_a_word(token):
                            end = mask_index + i
                            break
                    else:
                        end = len(sentence)
                    for i in range(start, end):
                        sentence[i] = self.tokenizer.mask_token
        return sentence

    def preprocess(self, texts: Iterable[Iterable[str]], labels: Iterable[Iterable[str]],
                   masks_proportion: float = 0.15, max_len: int = 512, sep: Union[Iterable[str], str] = None):
        """
        Preprocesses corpus for further usage by a model.
        :param texts: input texts
        :param labels: expected output texts
        :param masks_proportion: a proportion of text to mask
        :param max_len: the maximum number of tokens in a preprocessed text
        :param sep: separators that indicate the logical end of something (a sentence, of a thought, etc.) which could be used to divide a text into smaller pieces
        :return: preprocessed texts
        :rtype: transformers.tokenization_utils_base.BatchEncoding
        """

        def reshape(texts, labels):
            iterator = iter(labels)
            return [[next(iterator) for _ in text] for text in texts]

        self._define_token_of_word_beginning()
        shortened_texts = []
        shortened_labels = []
        true_max_len = max_len
        max_len -= 3  # we need to consider that special tokens will lengthen texts.
        for text, label in zip(texts, labels):
            text = "".join(text)
            tokenized_text = self.tokenizer.tokenize(text)
            tokenized_label = self.tokenizer.tokenize(label)
            if len(tokenized_text) > max_len:
                sep_splits = text.split(self.tokenizer.sep_token)
                tokenized_text = [self.tokenizer.tokenize(part) for part in sep_splits]
                text_splits = self._maximalize_length(tokenized_text, max_len=max_len)
                label_splits = reshape(sep_splits, labels)
                for text_split, label_split in zip(text_splits, label_splits):
                    if len(text_split) > max_len:
                        shorts = self.split_by_sep(text_split, sep=sep, max_len=max_len)
                        shorts = self._maximalize_length(shorts, max_len=max_len)
                        shorts_l = reshape(shorts, label_split)
                        for i, short in enumerate(shorts):
                            if len(short) > max_len:
                                cropped_text = self.cut(short, max_len=max_len)
                                cropped_label = reshape(cropped_text, shorts_l)
                                for short_text, short_label in zip(cropped_text, cropped_label):
                                    masked = self.mask_random(short_text, masks_proportion=masks_proportion)
                                    shortened_texts.append(self.tokenizer.convert_tokens_to_string(masked))
                                    shortened_labels.append(self.tokenizer.convert_tokens_to_string(short_label))
                            elif len(short) < 5 or not(sum([self._is_a_first_token_of_a_word(token) for token in short])):
                                shorts[i + 1] = short + shorts[i + 1]
                                shorts_l[i + 1] = shorts_l[i] + shorts_l[i + 1]
                                continue
                            else:
                                masked = self.mask_random(short, masks_proportion=masks_proportion)
                                shortened_texts.append(self.tokenizer.convert_tokens_to_string(masked))
                                shortened_labels.append(self.tokenizer.convert_tokens_to_string(shorts_l))
                    elif len(text_split) > 0:
                        masked = self.mask_random(text_split, masks_proportion=masks_proportion)
                        shortened_texts.append(self.tokenizer.convert_tokens_to_string(masked))
                        shortened_labels.append(self.tokenizer.convert_tokens_to_string(label_split))
            elif len(tokenized_text) > 0 :
                tokenized_text = list(filter(lambda a: a != self.tokenizer.sep_token, tokenized_text))
                masked = self.mask_random(tokenized_text, masks_proportion=masks_proportion)
                shortened_texts.append(self.tokenizer.convert_tokens_to_string(masked))
                tokenized_label = list(filter(lambda a: a != self.tokenizer.sep_token, tokenized_label))
                shortened_labels.append(self.tokenizer.convert_tokens_to_string(tokenized_label))
        result = self.tokenizer.batch_encode_plus(shortened_texts, padding="max_length", max_length=true_max_len, add_special_tokens=True, return_tensors="pt", return_attention_mask=True)
        result["labels"] = self.tokenizer.batch_encode_plus(shortened_labels, padding="max_length", max_length=true_max_len, add_special_tokens=True, return_tensors="pt", return_attention_mask=True).input_ids
        return result


class Corpus:
    """
    This class facilitates data manipulation for NER and MLM tasks.

    Attributes
    ----------
    path : str
        The path to the collection of raw texts and their annotations.
    model_name : str (default=None)
        The name of the model to use for tokenization and MLM.
    entities_types : list of str (default=[])
        The types of entities to mask.
    tokenizer : transformers.model
        The tokenizer used for preprocessing. Defaults to the standard tokenizer for the specified model.
    additional_entities : list of str
        Additional words to mask in the texts.
    raw_texts : dict
        A dictionary containing the raw texts with their corresponding IDs (file names) as keys.
    masked_texts : dict
        A dictionary containing the masked texts with their corresponding IDs as keys.
    entities_texts : dict
        A dictionary containing the entity-replaced texts with their corresponding IDs as keys.
    """

    def __init__(self, corpus_path, model_name=None, entities_types=(), additional_entities_paths=None, tokenizer=None,
                 texts_extension=".txt", annotations_extension=".ann", annotations_delimiter="\t"):
        self.path = corpus_path
        self.entities_types = [entity_type.upper() for entity_type in entities_types]
        self.model_name = model_name
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.model_name)
        self.additional_entities = []
        if additional_entities_paths:
            for path in additional_entities_paths:
                with open(path) as f:
                    self.additional_entities += f.readlines()
        self.raw_texts, self.masked_texts, self.entities_texts = self.__build_corpus__(texts_extension,
                                                                                       annotations_extension,
                                                                                       annotations_delimiter)

    def __build_corpus__(self, texts_extension=".txt", annotations_extension=".ann", annotations_delimiter="\t"):
        """
        Return three dictionaries with file names as keys.
        The first dictionary contains raw texts, the second dictionary contains masked texts
        and the third dictionnary contains the entity-replaced texts.
        """
        raw_texts = dict()
        masked_texts = dict()
        entities_texts = dict()
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(texts_extension):
                    file_id = file[:file.index(texts_extension)]
                    with open(os.path.join(root, file)) as f:
                        text = "".join(f.readlines())
                    raw_texts[file_id] = text
                    if self.entities_types:
                        entities = self.__get_entities__(
                            os.path.join(root, file.replace(texts_extension, annotations_extension)),
                            annotations_delimiter)
                        replaced = self.__mask_entities__(text, entities)
                        masked_texts[file_id], entities_texts[file_id] = replaced["mask"], replaced["NE"]
        if self.additional_entities:
            pre_masked_texts = masked_texts or raw_texts
            masked_texts = {text_id: self._naive_masking(text, self.additional_entities) for text_id, text in
                            pre_masked_texts.items()}
        return raw_texts, masked_texts, entities_texts

    def __get_entities__(self, file_path: str, annotations_delimiter: str):
        """
        Returns a dictionary containing an entity, its type and its placement in the text
        with the keys name, type, start and end.
        """
        with open(file_path) as f:
            entities = []
            for line in f:
                try:
                    i, info, name = line.strip().split(annotations_delimiter)
                    entity_type, start, end = info.split()
                    entities.append({"name": name, "type": entity_type.upper(), "start": int(start), "end": int(end)})
                except ValueError:
                    continue  # this error is raised while reading lines containing the annotators notes only
        return entities

    def __mask_entities__(self, text: str, entities: List[dict], mode=None):
        """
        Replaces entities of particular types in the input text with a "mask" token (if mode="mask"),
        with its entity type special token (if mode="NE"), or with both (if mode is not specified).
        Returns a dictionary containing the substitution mode as keys
        and the corresponding substitution results as values.
        """
        special_tokens = {"mask": lambda _: self.tokenizer.mask_token,
                          "NE": lambda entity: f'[{entity["type"].upper()}]'}
        if mode:
            special_tokens = {mode: special_tokens[mode]}
        result = {mode: text for mode in special_tokens.keys()}
        extensions = {mode: 0 for mode in special_tokens.keys()}
        # we'll mask entities in the order of their occurrence in the text
        for entity in sorted(entities, key=lambda x: x["start"]):
            if entity["type"] in self.entities_types:
                for mode, special_token in special_tokens.items():
                    masked_part = special_token(entity) * len(self.tokenizer.tokenize(entity["name"]))
                    result[mode] = result[mode][:entity["start"] + extensions[mode]] + masked_part + \
                                   result[mode][entity["end"] + extensions[mode]:]
                    extensions[mode] += len(masked_part) - len(entity["name"])
        return result

    def _naive_masking(self, text, entities):
        """Masks all the entities, including multi-word entities, in the text."""
        processed = [(entity.strip(), len(self.tokenizer.tokenize(entity))) for entity in entities]
        # We sort our list by entities length in order to mask the longest form of the same term
        processed.sort(key=lambda x: len(x), reverse=True)
        for entity in processed:
            try:
                text = re.sub(f"(?i){entity[0]}", self.tokenizer.mask_token * entity[1], text)
            except Exception:
                pass
        return text


class Model:
    """
    This class encapsulates a neural network model and related functionality for training, testing, and predicting.

    
    Attributes
    ----------
    device : str
        The device type where the model and other tensors will be stored during runtime.
    model_name : str
        A unique identifier for the model being used.
    module : transformers.model
        The neural network model being used for training, testing, and predicting.
    tokenizer : transformers.Tokenizer
        The tokenizer being used for preprocessing the text inputs.
        It may differ from the default tokenizer for the model.
    model : transformers.model or torch.nn.DataParallel
        A PyTorch DataParallel object that distributes the model across multiple GPUs, if available.
        If not, this is simply the module object.
    model_class : str
        The class name of the model architecture being used.
    repo : str
        The GitHub repository where models are stored.

    Methods
    -------
    train(sources, path_to_save, epochs=1000, patience = 5, lr=5e-5, optimizer=AdamW, save_after=129600, **log_settings)
        Train the model on a set of source data and save the trained model to the specified path.
    test(model_name_or_path, test_corpus_path, entities=None, print_output=False, masked_predictions_only=False)
        Test the model using the specified test data and return the evaluation metrics.
    predict(data, predictions_path=None, titles=None)
        Apply the trained model to new data and return the predictions.
    """
    def __init__(self, model_path: str = "", model_name: str = None, tokenizer: transformers.Tokenizer = None,
                 device: str = None, repo: str = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if repo:
            self.repo = git.Repo(repo)
        if os.path.isfile(model_path):
            self.module = torch.load(model_path, map_location=device)
        else:
            self.model_class = AutoConfig.from_pretrained(model_name).architectures[0]
            exec(f"from transformers import {self.model_class}")
            AutoModel = eval(self.model_class)
            self.module = AutoModel.from_pretrained(model_name)
        self.module.to(self.device)
        self.model = torch.nn.DataParallel(self.module).to(self.device) if "cuda" in self.device else self.module
        self.model_name = self.module.config._name_or_path
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.module.config._name_or_path)

    def _update_git(self, comment: str = ""):
        modified_files = [file.a_path for file in self.repo.index.diff(None)]
        self.repo.index.add(modified_files)
        self.repo.index.commit(comment)
        self.repo.remote().push()

    def _emergency_saving(self, path_to_save: str):
        torch.save(self.module, f"{path_to_save}/emergency.pt")
        if self.repo:
            self._update_git()

    def _calculate_accuracy(self, true: torch.Tensor, pred: torch.Tensor) -> float:
        correct = 0
        for pred_token, true_token in zip(pred, true):
            if pred_token == true_token:
                correct += 1
        return correct / len(true)

    def _load_data(self, inputs: torch.Tensor, i: int,
                   val_size: float = 0.2, batch_size: int = 16, shuffle: bool = True):
        """
        Splits data on batches and creates batch generators.
        :param inputs: The data to be split into batches.
        :param i: The total number of batches.
        :param val_size: The proportion of validation data to be used.
        :param batch_size: The number of samples in each batch.
        :param shuffle: Whether or not the data should be shuffled before splitting.
        :return: Generators for training and validation batches.
        """
        dataset = Dataset(inputs)
        len_dataset = len(dataset)
        val_size = int(len_dataset*val_size)
        val_start, val_end = val_size * (i-1), val_size * i
        val_ids = torch.arange(val_start,val_end)
        train_ids = torch.concat((torch.arange(0, val_start), torch.arange(val_end, len_dataset)), 0)
        train = torch.utils.data.Subset(dataset, train_ids)
        val = torch.utils.data.Subset(dataset, val_ids)
        num_workers = torch.cuda.device_count()
        trainload = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        valload = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        del dataset, train, val
        torch.cuda.empty_cache()
        return trainload, valload

    def _save_predictions(self, predicted_indices: torch.Tensor, texts_ids: List[str], save_path: str):
        """
        Saves model predictions to a file.
        :param predicted_indices: The predicted output of the model.
        :param texts_ids: A list of file names corresponding to the predictions.
        :param save_path: The file path where the predictions will be saved.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        texts = tokenizer.batch_decode(predicted_indices)
        del tokenizer
        for text_id, text in zip(texts_ids, texts):
            with open(f"{save_path}/{text_id}", "w") as f:
                f.write(text)
        print(f"Predictions are saved in {save_path}.")

    def _log(self, message: str, log_settings: dict[str, str]) -> str:
        if "path_to_save" in log_settings.keys():
            with open(log_settings["path_to_save"], "a") as blog:
                blog.write(f"\n{datetime.datetime.now()}\n{message}\n")
        return message

    def train(self, sources: List[str], path_to_save: str, entities_types: Iterable = (),
              additional_entities_paths: str = None, masks_proportion: float = 0, epochs: int = 1000,
              batch_size: int = 16, patience: int = 5, lr: float = 2e-5, optimizer=AdamW, save_after: int = 129600,
              **log_settings):
        """
        Train the model
        :param sources: list of str
            Paths to the training data.
        :param path_to_save: str
            Path to save the trained model.
        :param entities_types: list of str (None by default)
            Types of entities to mask.
        :param additional_entities_paths: str (None by default)
            Paths to additional entities to mask by exact matching.
        :param masks_proportion: float (0 by default)
            Proportion of entities to mask.
        :param epochs: int (1000 by default)
            The maximum number of epochs.
        :param batch_size: int (16 by default)
            The number of samples in each batch.
        :param patience: int (5 by default)
            The number of epochs after which the training will be stopped if the model does not improve its quality.
        :param lr: float (5e-5 by default)
            Learning rate.
        :param optimizer: PyTorch optimizer (AdamW by default)
        :param save_after: int (129600 (= 36 hours) be default)
            Time in seconds after which an additional saving will be done.
            During the training, the model will be saved if it outperforms previously saved model
            or if the epoch number is divisible by 100.
        :param log_settings: a path (str) for a txt file to log a training progress
        """
        emergency_saving = threading.Timer(save_after, torch.save, args=(self.module, f"{path_to_save}/emergency.pt"))
        emergency_saving.start()
        initial_patience = patience
        corpus, labels = [], []
        masks_proportion = masks_proportion or (0 if entities_types else 0.15)
        for source in sources:
            preprocessed_texts = Corpus(source, entities_types=entities_types,
                                        additional_entities_paths=additional_entities_paths,
                                        model_name=self.model_name, tokenizer=self.tokenizer)
            if masks_proportion:
                corpus += preprocessed_texts.masked_texts.values()
                labels += preprocessed_texts.raw_texts.values()
            else:
                corpus += preprocessed_texts.raw_texts.values()
                labels += preprocessed_texts.entities_texts.values()
        if not corpus: corpus = labels
        preprocessor = Preprocessor(AutoTokenizer.from_pretrained(self.model_name))
        data = preprocessor.preprocess(texts=corpus, labels=labels, masks_proportion=masks_proportion)
        optimizer = optimizer(self.model.parameters(), lr=lr)
        self.min_val_loss = float("inf")
        for e in range(1, epochs+1):
            train_data, val_data = self._load_data(data, i=e % 5+1, batch_size=batch_size)
            torch.cuda.empty_cache()
            self._epoch(train_data, optimizer)
            val_loss = self._validate(val_data)
            message = f"Model {self.model_name} on {sources}.\n Epoch {e} of {epochs}.\nValidation loss : {val_loss}"
            print(self._log(message, log_settings))
            if e % 100 == 0: torch.save(self.module, f"{path_to_save}/{e}.pt")
            if val_loss < self.min_val_loss:
                patience = initial_patience
                self.min_val_loss = val_loss
                torch.save(self.module, f"{path_to_save}/best.pt")
            else:
                patience -= 1
                if patience == 0: break
        torch.save(self.module, f"{path_to_save}/last.pt")
        print(self._log(f"Model {self.model_name} on {sources} is ready. Loss - {self.min_val_loss}", log_settings))
        emergency_saving.cancel()
        if self.repo: self._update_git()

    def _epoch(self, train_data: torch.utils.data.DataLoader, optimizer):
        loop = tqdm(train_data, leave=True)
        self.model.train()
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.mean().backward()
            optimizer.step()
            loop.set_postfix(loss=loss)
            del batch, outputs, loss, input_ids, attention_mask, labels
            torch.cuda.empty_cache()
        del loop
        torch.cuda.empty_cache()

    def _validate(self, val_data: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_data:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.mean().item()
                del batch, input_ids, attention_mask, labels, outputs
                torch.cuda.empty_cache()
            return val_loss/len(val_data)

    def test(self, test_corpus_path: str, predict_random: bool = True, entities_types: Iterable = (),
             batch_size: int = 16, print_output: bool = False,
             masked_predictions_only: bool = True, predictions_path: str = None) -> (float, float):
        """
        Test the model and calculate its performance metrics.
        :param test_corpus_path: str
            A path to the test data, which consists of texts and their corresponding gold annotations.
        :param predict_random: bool
            If True, the language model will be tested for random masked token prediction, if False, it will be tested for masked entity prediction.
        :param entities_types: list of str
            A list of entity types to mask in the test corpus.
        :param batch_size: int (16 by default)
            The number of samples in each batch.
        :param print_output: bool
            If True, the accuracy and the perplexity of the model will be printed.
        :param masked_predictions_only: bool
            If True, only the masked predictions will be taken into consideration while calculating the score.
        :param predictions_path: str (optional)
            A path to save the predictions in text format.
        :return: A tuple containing accuracy and perplexity scores.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        test_corpus = Corpus(test_corpus_path, model_name=self.model_name, entities_types=entities_types)
        titles = test_corpus.raw_texts.keys()
        preprocessor = Preprocessor(tokenizer)
        if predict_random:
            masks_proportion = 0 if entities_types else 0.15
            data = preprocessor.preprocess(texts=test_corpus.masked_texts.values() or test_corpus.raw_texts.values() , labels=test_corpus.raw_texts.values(), masks_proportion=masks_proportion)
        else:
            data = preprocessor.preprocess(texts=test_corpus.raw_texts.values(), labels=test_corpus.entities_texts.values())
        answers = data.labels.clone().detach()
        self.mask_token = preprocessor.tokenizer.mask_token_id
        del preprocessor, test_corpus
        batches = torch.utils.data.DataLoader(Dataset(data), batch_size=batch_size, num_workers=torch.cuda.device_count())
        del data
        torch.cuda.empty_cache()
        self.test_loss, self.accuracy, shift = 0, 0, 0
        for i, batch in enumerate(batches):
            start, end = batch_size*i, batch_size*(i+1)
            outputs = self.predict(batch, masked_predictions_only=masked_predictions_only).to(self.device)
            if predictions_path: self._save_predictions(outputs, titles, predictions_path)
            true = answers[start:end][batch["input_ids"] == self.mask_token]
            pred = outputs[batch["input_ids"] == self.mask_token]
            if true.numel(): self.accuracy +=  self._calculate_accuracy(true=true, pred=pred)
            else: shift += 1
            del outputs, true, pred, batch
            torch.cuda.empty_cache()
        self.accuracy /= (i+1-shift)
        self.test_loss /= (i+1-shift)
        self.perplexity = torch.exp(torch.tensor(self.test_loss))
        if print_output: print(f"The quality of a {self.model_name} model on a {test_corpus_path.split('/')[-1]} corpus is {self.accuracy} with the perplexity of {self.perplexity}.")
        return self.accuracy, self.perplexity

    def predict(self, data, masked_predictions_only: bool) -> torch.Tensor:
        """
        Apply the model to input data and obtain its predictions.
        :param data: Preprocessed data.
        :param masked_predictions_only: If True, only predictions for masked entities will be returned.
        :return: Predictions in PyTorch tensor format.
        """
        self.model.eval()
        with torch.no_grad():
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            labels = data["labels"].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            self.test_loss += outputs.loss.mean()
            logits = outputs.logits
            del outputs
            torch.cuda.empty_cache()
        softmax = torch.nn.functional.softmax(logits, dim=-1)
        predicted_indeces = torch.argmax(softmax, dim=-1)
        del softmax, logits
        torch.cuda.empty_cache()
        if masked_predictions_only: predicted_indeces = torch.where(input_ids == self.mask_token, predicted_indeces, input_ids)
        return predicted_indeces


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings): self.encodings = encodings
    def __getitem__(self, idx): return {key: val[idx].clone().detach() for key, val in self.encodings.items()}
    def __len__(self): return len(self.encodings.input_ids)