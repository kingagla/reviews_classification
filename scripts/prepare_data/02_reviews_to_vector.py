import pickle
import time
import os
import pandas as pd
import torch
from transformers import RobertaModel, PreTrainedTokenizerFast
from scripts.utils import lemmatize_text


def reviews_to_vector(reviews, tokenizer, model, lemmatize=False):
    # save each opinion as vector
    for i, review in enumerate(reviews['Opinion']):
        try:
            start_time = time.time()
            # skip opinions if is in the middle of OpiConnect process (then text of opinion is hidden)
            if 'Opinia jest w trakcie OpiConnect' in review:
                continue
            # remove information that opinion was in OpiConnect before
            elif 'OpiConnect' in review:
                review = review.replace('Opinia była przedmiotem dialogu w ramach procesu OpiConnect. \
                Strony nie osiągnęły porozumienia.', '')
                review = review.replace('Opinia była przedmiotem dialogu w OpiConnect. Strony osiągnęły porozumienie.')

            if lemmatize:
                review = lemmatize_text(review)

            # text to vector
            input_ = tokenizer.encode(review)
            output = model(torch.tensor([input_]))[1]

            # Save each vector as pickle file
            if lemmatize:
                with open(f'../../data/roberta_pickles_lem/{i}.pkl', 'wb') as f:
                    pickle.dump(output, f)
            else:

                with open(f'../../data/roberta_pickles/{i}.pkl', 'wb') as f:
                    pickle.dump(output, f)

            print(i, time.time() - start_time)
        except IndexError as e:
            print(e)


if __name__ == '__main__':
    # load reviews and model
    model_dir = "../../roberta"
    reviews = pd.read_csv('../../data/reviews.csv')
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
    model = RobertaModel.from_pretrained(model_dir, output_hidden_states=True)
    reviews_to_vector(reviews, tokenizer, model, lemmatize=True)
