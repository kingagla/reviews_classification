import re
import matplotlib.pyplot as plt
import morfeusz2
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from scripts.utils import lemmatize_text


def save_wordcloud(text, name, stopwords):
    # generate word cloud
    cloud = WordCloud(stopwords=stopwords).generate(text)

    # plot and save figure
    plt.figure()
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    if stopwords is None:
        plt.savefig(f'../../plots/with_stopwords/cloud_{name}.png', dpi=500, format='png')
    else:
        plt.savefig(f'../../plots/without_stopwords/cloud_{name}.png', dpi=500, format='png')


def plot_word_freq(text, name, stopwords, n=30):
    # remove stopwords if obligatory
    if stopwords is None:
        text = text.split()
    else:
        text = [word for word in text.split() if word not in stop_words.values.flatten()]

    # create word counter
    words = Counter()
    words.update(text)

    # create df containing word frequency
    popular_words = list(map(lambda x: x[0], words.most_common()))
    amount = list(map(lambda x: x[1], words.most_common()))
    df = pd.DataFrame({'Word': popular_words, 'Amount': amount})

    # plot barplot
    plt.figure(figsize=(20, 10))
    sns.set(font_scale=1.5)
    sns.barplot('Amount', 'Word', data=df.iloc[:n])
    if stopwords is None:
        plt.savefig(f'../../plots/with_stopwords/bar_{name}.png', dpi=500, format='png')
    else:
        plt.savefig(f'../../plots/without_stopwords/bar_{name}.png', dpi=500, format='png')


def main():
    global reviews, stop_words, remove_stopwords
    # generate plot for each type of reviews (pos, neg, neu)
    for inf, file_name in zip(['pos', 'neg', 'neu'], ['positive', 'negative', 'neutral']):
        # choose opinion type
        opinions = reviews['Opinion'][reviews['Information'] == inf]
        # remove punctuation marks for word cloud
        opinions = opinions.apply(lambda x: re.sub('[^a-żA-Ż]', ' ', x))
        # join opinions in one long text (in lowercase)
        opinions = " ".join(opinions.values)
        opinions = opinions.lower()
        # lemmatize text
        opinions = lemmatize_text(opinions)
        # create and save word cloud and barplots
        if remove_stopwords:
            save_wordcloud(opinions, file_name, stop_words.values.flatten())
            plot_word_freq(opinions, file_name, stop_words.values.flatten())
        else:
            save_wordcloud(opinions, file_name, None)
            plot_word_freq(opinions, file_name, None)


if __name__ == '__main__':
    # load reviews and stopwords
    reviews = pd.read_csv('../../data/reviews.csv')
    stop_words = pd.read_csv('../../data/polish_stopwords.txt', names=['stopwords'], dtype={'stopwords': str})
    remove_stopwords = True

    for remove_stopwords in [True, False]:
        main()