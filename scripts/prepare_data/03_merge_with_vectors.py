import pandas as pd


def review_vector_merge(reviews):
    vectors = []
    for i in range(reviews.shape[0]):
        try:
            vec = pd.read_pickle(f'../../data/roberta_pickles/{i}.pkl')
            vectors.append((i, vec))
        except Exception as e:
            print(e)

    vals = [x[1] for x in vectors]
    ids = [x[0] for x in vectors]

    df = pd.DataFrame(vals, columns=[f'Vec_{i}' for i in range(1024)], index=ids)
    reviews = reviews.merge(df, left_index=True, right_index=True)
    reviews.to_pickle('../../data/reviews_with_vec.pickle')


if __name__ == '__main__':
    reviews = pd.read_csv('../../data/reviews.csv')
    review_vector_merge(reviews)
