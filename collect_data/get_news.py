import os
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
import logging
import re
from datetime import datetime, timedelta
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService

import psycopg2
from psycopg2 import sql
from psycopg2.errors import UniqueViolation

HSI_COMPONENTS = [
    "0001.HK",
    # "0002.HK", "0003.HK", "0005.HK", "0006.HK",
    # "0011.HK", "0012.HK", "0016.HK", "0017.HK", "0027.HK",
    # "0066.HK", "0101.HK", "0175.HK", "0241.HK", "0285.HK",
    # "0288.HK", "0291.HK", "0316.HK", "0322.HK", "0386.HK",
    # "0388.HK", "0669.HK", "0688.HK", "0700.HK", "0762.HK",
    # "0823.HK", "0836.HK", "0857.HK", "0868.HK", "0881.HK",
    # "0883.HK", "0939.HK", "0941.HK", "0960.HK", "0968.HK",
    # "0981.HK", "0992.HK", "1024.HK", "1038.HK", "1044.HK",
    # "1088.HK", "1093.HK", "1099.HK", "1109.HK", "1113.HK",
    # "1177.HK", "1209.HK", "1211.HK", "1299.HK", "1378.HK",
    # "1398.HK", "1810.HK", "1876.HK", "1928.HK", "1929.HK",
    # "1997.HK", "2015.HK", "2020.HK", "2057.HK", "2269.HK",
    # "2313.HK", "2318.HK", "2319.HK", "2331.HK", "2359.HK",
    # "2382.HK", "2388.HK", "2628.HK", "2688.HK", "2899.HK",
    # "3690.HK", "3692.HK", "3968.HK", "3988.HK", "6618.HK",
    # "6690.HK", "6862.HK", "9618.HK", "9633.HK", "9888.HK",
    # "9901.HK", "9961.HK", "9988.HK", "9999.HK",
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

YAHOO_FINANCE_NEWS_URL_TEMPLATE = "https://hk.finance.yahoo.com/quote/{}/news/"
# OUTPUT_CSV_DIR = "./stock_news_data/"
#
# if not os.path.exists(OUTPUT_CSV_DIR):
#     os.makedirs(OUTPUT_CSV_DIR)

DB_CONFIG = {
    'host': 'xxxx',
    'dbname': 'xxxx',
    'user': 'xxxx',
    'password': 'xxxx',
    'port': ''
}

def scroll_down_page(driver, scroll_count=5, scroll_pause_time=5):
    last_height = driver.execute_script("return document.body.scrollHeight")
    for i in range(scroll_count):
        logging.info(f"Scrolling page ({i + 1}/{scroll_count})...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            logging.info("Page reached bottom or no more content to load.")
            break
        last_height = new_height

# 不重要了，可以忽视
def parse_datetime_from_text(raw_text):
    now = datetime.now()
    published_dt = None

    raw_text = raw_text.strip()

    hours_ago_match = re.search(r'(\d+)\s*小时前', raw_text)
    minutes_ago_match = re.search(r'(\d+)\s*分钟前', raw_text)

    if hours_ago_match:
        hours = int(hours_ago_match.group(1))
        published_dt = now - timedelta(hours=hours)
    elif minutes_ago_match:
        minutes = int(minutes_ago_match.group(1))
        published_dt = now - timedelta(minutes=minutes)
    elif raw_text.startswith('今天'):
        time_str = raw_text.replace('今天', '').strip()
        try:
            time_obj = datetime.strptime(time_str, '%H:%M').time()
            published_dt = now.replace(hour=time_obj.hour, minute=time_obj.minute, second=0, microsecond=0)
            if published_dt > now:
                published_dt -= timedelta(days=1)
        except ValueError:
            pass
    elif raw_text == '昨天':
        published_dt = now - timedelta(days=1)
        published_dt = published_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif re.match(r'\d{1,2}月\d{1,2}日', raw_text):
        try:
            month, day = map(int, re.findall(r'(\d+)', raw_text))
            year = now.year
            if month > now.month:
                year -= 1
            published_dt = datetime(year, month, day, 0, 0, 0)
        except ValueError:
            pass
    elif re.match(r'\d{4}年\d{1,2}月\d{1,2}日', raw_text):
        try:
            year, month, day = map(int, re.findall(r'(\d+)', raw_text))
            published_dt = datetime(year, month, day, 0, 0, 0)
        except ValueError:
            pass
    else:
        logging.debug(f"Could not parse raw_text '{raw_text}' into a specific datetime format.")

    if published_dt:
        return published_dt.isoformat()
    return None

def create_news_table():
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS yahoo_finance_news (
                uuid BIGINT NOT NULL,
                stock_code VARCHAR(10) NOT NULL,
                title TEXT,
                publisher VARCHAR(255),
                link TEXT,
                published_raw_text VARCHAR(255),
                published_datetime TIMESTAMP WITH TIME ZONE,
                full_text TEXT,
                main_image_url TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                PRIMARY KEY (uuid, stock_code)
            );
        """)
        conn.commit()
        logging.info("Table 'yahoo_finance_news' checked/created successfully with composite primary key.")
    except Exception as e:
        logging.error(f"Error creating table: {e}")
    finally:
        if conn:
            conn.close()

def insert_news_data(news_data_list):
    if not news_data_list:
        logging.info("No news data to insert into database.")
        return

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        insert_query = sql.SQL("""
            INSERT INTO yahoo_finance_news (
                uuid, stock_code, title, publisher, link, published_raw_text,
                published_datetime, full_text, main_image_url
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) ON CONFLICT (uuid, stock_code) DO NOTHING; -- 冲突策略改为复合主键
        """)

        data_to_insert = []
        for item in news_data_list:
            data_to_insert.append((
                item.get('UUID'),
                item.get('Stock_code'),
                item.get('Title'),
                item.get('Publisher'),
                item.get('Link'),
                item.get('Published_Raw_Text'),
                item.get('Published_Datetime'),
                item.get('Full_Text'),
                item.get('Main_Image_URL')
            ))

        cur.executemany(insert_query, data_to_insert)
        conn.commit()
        logging.info(f"Successfully inserted {cur.rowcount} new news items into PostgreSQL.")

    except UniqueViolation as e:
        logging.warning(f"Unique constraint violation during insertion (some items might be duplicates): {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        logging.error(f"Error inserting news data into PostgreSQL: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            cur.close()
            conn.close()

def get_article_full_text(driver, article_url):
    full_text = ""
    publisher = "N/A"
    published_datetime_standard = None
    main_image_url = None

    original_window = driver.current_window_handle

    try:
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[-1])

        logging.info(f"Visiting article URL for full content: {article_url}")
        driver.get(article_url)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article.gridLayout.yf-1xsmqro"))
        )
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        try:
            byline_div = soup.find('div', class_='byline yf-1k5w6kz')
            if byline_div:
                author_div = byline_div.find('div', class_='byline-attr-author yf-1k5w6kz')
                if author_div:
                    publisher = author_div.get_text(strip=True)

                time_tag = byline_div.find('time', class_='byline-attr-meta-time')
                if time_tag:
                    raw_datetime_str = time_tag.get('datetime')
                    if raw_datetime_str:
                        try:
                            published_dt_obj = datetime.fromisoformat(raw_datetime_str.replace('Z', '+00:00'))
                            published_datetime_standard = published_dt_obj.strftime('%Y-%m-%d %H:%M:%S')
                        except ValueError as ve:
                            logging.warning(f"Error parsing datetime string '{raw_datetime_str}': {ve}")
                            published_datetime_standard = raw_datetime_str

            logging.info(f"Extracted Publisher: '{publisher}', Standard Datetime: '{published_datetime_standard}'")

        except Exception as e:
            logging.error(f"Error extracting publisher/time for {article_url}: {e}")

        image_container_element = soup.find('div', class_='image-container yf-1vr77wf')
        if image_container_element:
            img_tag = image_container_element.find('img')
            if img_tag and img_tag.get('src'):
                main_image_url = img_tag['src']
                logging.info(f"Extracted main image URL: {main_image_url[:50]}...")

            image_container_element.decompose()

        content_body_wrap = soup.find('div', class_='body-wrap yf-40hgrf')
        if content_body_wrap:
            for unwanted_tag in content_body_wrap.find_all(
                    ['blockquote', 'figure', 'aside', 'script', 'style', 'div', 'span'],
                    class_=lambda x: x and (
                            'ad-unit' in x or 'ad-container' in x or 'related-articles' in x or 'mod-recirc' in x or 'ad-placeholder' in x)):
                unwanted_tag.decompose()

            paragraphs = content_body_wrap.find_all('p', class_='yf-1090901')

            temp_full_text_parts = []
            if paragraphs:
                for p in paragraphs:
                    para_text = p.get_text(strip=True)
                    if para_text.startswith('#') and len(para_text.split()) < 10:
                        logging.debug(f"Skipping potential tag line: {para_text[:50]}...")
                        continue

                    cleaned_para_text = re.sub(r'(\s*<br\s*/?>\s*)+', '\n', para_text, flags=re.IGNORECASE)

                    if cleaned_para_text:
                        temp_full_text_parts.append(cleaned_para_text)

                full_text = "\n".join(temp_full_text_parts)

            else:
                full_text = content_body_wrap.get_text(separator="\n", strip=True)

                lines = full_text.split('\n')
                cleaned_lines = []
                for line in lines:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    if line_stripped.startswith('#') and len(line_stripped.split()) < 10:
                        logging.debug(f"Skipping potential tag line in general text: {line_stripped[:50]}...")
                        continue
                    cleaned_lines.append(line_stripped)
                full_text = "\n".join(cleaned_lines)

            logging.info(f"Successfully extracted and cleaned full text for: {article_url[:50]}...")

        else:
            logging.warning(f"Could not find main content wrapper (body-wrap) for article: {article_url}")

    except Exception as e:
        logging.error(f"Error fetching full details for {article_url}: {e}")
    finally:
        driver.close()
        driver.switch_to.window(original_window)
        time.sleep(1)

    return full_text, publisher, published_datetime_standard, main_image_url

def parse_news_elements(driver, stock_code):
    news_items = []
    try:
        news_section_xpath = ("//ul[contains(@class, 'stream-items')]/li/section[@data-testid='storyitem']"
                              "/div[contains(@class, 'content')]")
        all_news_sections = driver.find_elements(By.XPATH, news_section_xpath)

        logging.info(f"Found {len(all_news_sections)} potential news links on the page.")

        parsed_count = 0
        for i, section_element in enumerate(all_news_sections):
            logging.info(f"Attempting to parse news item {i + 1}/{len(all_news_sections)} for {stock_code}...")

            title = None
            href = None
            publisher = "Unknown Publisher"
            published_raw_text = "Null"
            published_datetime = None
            article_full_text = ""
            main_image_url = None
            uuid = None

            try:
                link_element = section_element.find_element(By.XPATH, ".//a[contains(@class, 'subtle-link') and "
                                                                      "contains(@class, 'title')]")
                title = link_element.get_attribute('aria-label') or \
                        link_element.get_attribute('title') or \
                        link_element.text.strip()
                href = link_element.get_attribute('href')
                uuid = hash(href) if href else None

            except NoSuchElementException:
                logging.warning(
                    f"Could not find main news link/title in section {i + 1}. Skipping this item for {stock_code}.")
                continue

            if not title:
                logging.info(f"Skipping empty title for news item {i + 1}.")
                continue

            if not href:
                logging.info(f"Skipping empty link for news item {i + 1}.")
                continue

            if href and "yahoo.com/news/" in href:
                article_full_text, detail_publisher, detail_published_datetime, main_image_url = get_article_full_text(driver, href)
                if detail_publisher and detail_publisher != "N/A":
                    publisher = detail_publisher
                if detail_published_datetime:
                    published_datetime = detail_published_datetime
            else:
                logging.info(f"Skipping full text extraction and detail parsing for external link: {href[:50]}...")

                try:
                    footer_div = section_element.find_element(By.XPATH, "./div[contains(@class, 'footer')]")
                    publishing_div = footer_div.find_element(By.XPATH, "./div[contains(@class, 'publishing')]")
                    full_published_raw_text = publishing_div.text.strip()
                    logging.info(f"DEBUG: full_raw_text_from_element for {href[:50]} : '{full_published_raw_text}'")
                    if '•' in full_published_raw_text:
                        published_raw_text = full_published_raw_text.split('•')[-1].strip()
                        logging.info(f"DEBUG: published_raw_text after split for {href[:50]} : '{published_raw_text}'")
                    else:
                        published_raw_text = full_published_raw_text
                        logging.info(f"DEBUG: published_raw_text (no split) for {href[:50]} : '{published_raw_text}'")

                    published_datetime = parse_datetime_from_text(published_raw_text)
                    publisher = "Unknown Publisher"
                except NoSuchElementException:
                    logging.warning(f"Could not find list page time info for external/skipped link {href[:50]}...")
                except Exception as e:
                    logging.warning(
                            f"Error extracting list page raw time for external/skipped link {href[:50]}: {e}")

            news_items.append({
                'Stock_code': stock_code,
                'UUID': uuid,
                'Title': title,
                'Publisher': publisher,
                'Link': href,
                'Published_Raw_Text': published_raw_text,
                'Published_Datetime': published_datetime,
                'Full_Text': article_full_text,
                'Main_Image_URL': main_image_url
            })
            parsed_count += 1

    except NoSuchElementException:
        logging.warning(
            f"News list section elements matching specified XPath not found on {stock_code} page. Page structure might have changed or no news available. XPath: {news_section_xpath}")
    except Exception as e:
        logging.error(f"Unexpected error occurred while parsing news elements for {stock_code}: {e}")

    return news_items

def crawl_yahoo_finance_news():
    all_news_data = []

    chrome_options = webdriver.ChromeOptions()

    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    prefs = {
        'profile.default_content_settings.popups': 0,
    }

    chrome_options.add_experimental_option('prefs', prefs)

    try:
        # chrome_service = ChromeService(executable_path='./chromedriver')
        chrome_service = ChromeService(ChromeDriverManager().install())

        driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
        driver.implicitly_wait(10)

        logging.info("Starting Yahoo Finance news crawling...")

        stock_codes = HSI_COMPONENTS
        if not stock_codes:
            logging.error("No stock_code to crawl. Please check.")
            driver.quit()
            return

        for code in stock_codes:
            logging.info(f"\n--- Crawling news for: {code} ---")
            news_url = YAHOO_FINANCE_NEWS_URL_TEMPLATE.format(code)

            try:
                driver.get(news_url)
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.XPATH,
                                                "//ul[contains(@class, 'stream-items')]/li/section[@data-testid='storyitem']/div[contains(@class, 'content')]/a[contains(@class, 'subtle-link') and contains(@class, 'titles')]"))
                )

                scroll_down_page(driver, scroll_count=3, scroll_pause_time=5)

                current_stock_news = parse_news_elements(driver, code)
                if current_stock_news:
                    insert_news_data(current_stock_news)
                    logging.info(f"Collected and inserted {len(current_stock_news)} news items for {code}.")
                else:
                    logging.info(f"No news data collected for {code}.")

                time.sleep(3)

            except TimeoutException:
                logging.warning(f"Timeout or news elements not found for {code} page.")
            except Exception as e:
                logging.error(f"An error occurred during processing {code}: {e}")

    except WebDriverException as e:
        logging.error(f"Failed to start Chrome WebDriver or encountered a WebDriver error during crawling. Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during crawling: {e}")
    finally:
        if driver:
            driver.quit()

    # if all_news_data:
    #     df = pd.DataFrame(all_news_data)
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     output_csv_path = os.path.join(OUTPUT_CSV_DIR, f"yahoo_finance_news_{timestamp}.csv")
    #
    #     df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    #     logging.info(f"\nAll stock news saved to: {output_csv_path}")
    #     logging.info(f"Total {len(all_news_data)} news items collected.")
        # insert_news_data(all_news_data)

    logging.info("Crawler finished. Data has been processed to PostgreSQL database.")
    # else:
    #     logging.info("\nNo news data collected.")

if __name__ == "__main__":
    crawl_yahoo_finance_news()
