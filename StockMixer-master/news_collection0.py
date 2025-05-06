import yfinance as yf
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# {stock:{time:[news0,news1,...]}}

start_date = "2015-03-01"
end_date = "2025-03-24"
hk_stock_codes = ["0700.HK", "0001.HK", "0002.HK", "0003.HK"]
news = yf.Search("Google", news_count=10000).news
print(news)
print(len(news))

# stock_data_dict = {}
#
# for ticker in hk_stock_codes:
#     try:
#         stock_data = yf.download(ticker, start=start_date, end=end_date)
#         stock_data_dict[ticker] = stock_data
#     except Exception as e:
#         print(f"Failed to download data for {ticker}: {e}")
#
#
# with open("hk_stock_data.pkl", "wb") as f:
#     pickle.dump(stock_data_dict, f)

ticker = "0700.HK"
stock = yf.Ticker(ticker)
news_data = stock.news
print(news_data)
print(len(news_data))
# print(type(news_data))
# a = news_data[0]['content']
# print(a['summary'])
# for key in a.keys():
#     print(key)
#     print(a[key])
# for news in news_data:
#     title = news['content']['title']
#     link = news['content']['canonicalUrl']['url']
#     pub_date = news['content']['pubDate']
#     publish_time = datetime.strptime(pub_date, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M:%S')
#     print(f"title: {title}")
#     print(f"link: {link}")
#     print(f"time: {publish_time}\n")

# news_link = r'https://finance.yahoo.com/news/tesla-tsla-gains-analyst-upgrade-133754285.html'
# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
# driver.get(news_link)
# try:
#     news_content_element = driver.find_element(By.CSS_SELECTOR,'#nimbus-app > section > section > section > article > div > div.article-wrap.no-bb > div.body-wrap.yf-40hgrf > div.body.yf-1ujgn8c > div.atoms-wrapper')
#     print(news_content_element)
#     a = news_content_element.text
#     print(a)
#     # soup = BeautifulSoup(news_content, 'html.parser')
#     # formatted_news = soup.get_text(separator="\n", strip=True)
#     # print("news content:")
#     # print(formatted_news)
# except Exception as e:
#     print("error!")
# finally:
#     driver.quit()

