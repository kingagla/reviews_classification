import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# define function to save word cloud
def save_wordcloud(text, name, stopwords):
    # generate word cloud
    cloud = WordCloud(stopwords=stopwords).generate(text)

    # plot and save figure
    plt.figure()
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(f'./plots/{name}_cloud.png', dpi=500, format='png')
    print('File saved')


if __name__ == '__main__':
    # load reviews and stopwords
    reviews = pd.read_csv('./data/reviews.csv', sep=';', names=['star', 'inf', 'opinion'], engine='python',
                          encoding='utf-8')
    stop_words = pd.read_csv('./data/polish_stopwords.txt', names=['stopwords'], dtype={'stopwords': str})

    # generate plot for each type of reviews (pos, neg, neu)
    for inf, file_name in zip(['pos', 'neg', 'neu'], ['positive', 'negative', 'neutral']):
        # choose opinion type
        opinions = reviews['opinion'][reviews['inf'] == inf]
        # remove punctuation marks for word cloud
        opinions = opinions.apply(lambda x: re.sub('[^a-żA-Ż]', ' ', x))
        # join opinions in one long text (in lowercase)
        opinions = " ".join(opinions.values)
        opinions = opinions.lower()
        # create and save word cloud
        save_wordcloud(opinions, file_name, stop_words.values.flatten())
