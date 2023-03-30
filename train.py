import argparse

from utils import Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fine-tune language models')
    parser.add_argument('--data', type=str, help='Textual data name. While indicating several types, please, use the "_" symbol as a separator.')
    parser.add_argument('--model_name', type=str, nargs="?", default="", help='Model name.')
    parser.add_argument('--model_path', type=str, nargs="?", default="", help='Model path.')
    parser.add_argument('--path_to_save', type=str, nargs="?", default="models", help='Path to save model checkpoints.')
    parser.add_argument('--entities', type=str, nargs="?", default="", help='Entities types to predict. While indicating several types, please, use the "_" symbol as a separator.')
    parser.add_argument('--lexicons', type=str, nargs="?", default="", help='Lexicon of words to mask. While indicating several paths, please, use the "_" symbol as a separator.')
    args = parser.parse_args()
    sources = args.data.split('_')
    entities_types_to_predict = list(filter(lambda x:x, args.entities.split("_")))
    additional_entities_paths=args.lexicons.split("_")
    model = Model(model_name=args.model_name, model_path=args.model_path)
    model.train(sources, path_to_save=args.path_to_save, entities_types=entities_types_to_predict,
                  additional_entities_paths=additional_entities_paths, masks_proportion=0.15,
                  epochs=100, patience=10, save_after=169200)
