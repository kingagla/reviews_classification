import csv
import requests
from bs4 import BeautifulSoup
import time


def get_review(url):
    # open csv
    file = open('./data/reviews.csv', 'a', encoding='utf-8')
    # define settings for csv
    writer = csv.writer(file, delimiter=';', lineterminator='\n')
    # get page as xml file
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    revs_text = [item.text.replace('\n', ' ').replace('\r',' ') for item in soup.find_all('div', class_='revz_txt')]
    stars = (float(item.text.split('/')[0]) for item in soup.find_all('span', class_='review_badge'))
    revs = (item.get('class')[1] for item in soup.find_all('span', class_='review_badge'))

    # write rows to document
    for i, (star, rev, text) in enumerate(zip(stars, revs, revs_text)):
        writer.writerow([star, rev, text])
    file.close()
    return i


if __name__ == '__main__':
    # download opinions - limiting to 20 000 due to my computer power
    n = 0
    page_num = 9073
    max_opinions = 20000
    while n <= max_opinions:
        start_time = time.time()
        # define web page for scrapping
        page = f'https://www.opineo.pl/opinie/dhl-com-pl/{page_num}#opinie'
        n += get_review(page)
        page_num -= 1
        print(n, page_num, time.time()-start_time)


