import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List, Dict
import logging
import urllib.parse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SHLScraper:
    def __init__(self):
        self.base_url = "https://www.shl.com/solutions/products/product-catalog/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_page_content(self, start: int, type_num: int) -> str:
        params = {'start': start, 'type': type_num}
        try:
            response = requests.get(self.base_url, params=params, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logging.error(f"Error fetching page: {e}")
            return ""

    def check_yes_no(self, cell) -> str:
        yes_span = cell.find('span', class_='catalogue__circle -yes')
        no_span = cell.find('span', class_='catalogue__circle -no')
        if yes_span:
            return "Yes"
        elif no_span:
            return "No"
        return ""

    def get_test_link(self, cell) -> str:
        link = cell.find('a')
        if link and 'href' in link.attrs:
            return link['href']
        return ""

    def get_test_description_and_more(self, test_link: str) -> Dict:
        result = {
            'Description': "",
            'Job Levels': "",
            'Assessment Length': ""
        }

        if not test_link:
            return result

        # Construct full URL
        if test_link.startswith('/'):
            test_link = urllib.parse.urljoin("https://www.shl.com", test_link)

        try:
            logging.info(f"Fetching assessment detail page: {test_link}")
            response = requests.get(test_link, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Parse each row in the training calendar section
            rows = soup.find_all('div', class_='product-catalogue-training-calendar__row typ')
            for row in rows:
                heading = row.find('h4')
                paragraph = row.find('p')
                if heading and paragraph:
                    title = heading.get_text(strip=True).lower()
                    value = paragraph.get_text(strip=True)

                    if 'description' in title:
                        result['Description'] = value
                    elif 'job level' in title:
                        result['Job Levels'] = value
                    elif 'assessment length' in title:
                        result['Assessment Length'] = value

            time.sleep(1)
        except requests.RequestException as e:
            logging.error(f"Error fetching full details from {test_link}: {e}")

        return result

    def extract_table_data(self, html_content: str, max_limit: int = None) -> List[Dict]:
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')

        all_data = []
        for table in tables:
            rows = table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                # Only break if max_limit is defined and reached
                if max_limit is not None and len(all_data) >= max_limit:
                    return all_data

                cols = row.find_all('td')
                if len(cols) >= 4:
                    # 🆕 Ensure full test link
                    test_link = urllib.parse.urljoin("https://www.shl.com", self.get_test_link(cols[0]))
                    test_name = cols[0].get_text(strip=True)
                    remote_testing = self.check_yes_no(cols[1])
                    adaptive_irt = self.check_yes_no(cols[2])
                    test_type = cols[3].get_text(strip=True)

                    detail_data = self.get_test_description_and_more(test_link)

                    data = {
                        'Test Name': test_name,
                        'Test Link': test_link,  # full URL now
                        'Remote Testing': remote_testing,
                        'Adaptive/IRT': adaptive_irt,
                        'Test Type': test_type,
                        'Description': detail_data['Description'],
                        'Job Levels': detail_data['Job Levels'],
                        'Assessment Length': detail_data['Assessment Length']
                    }

                    all_data.append(data)
        return all_data

    def scrape_all_tables(self, max_pages: int = 100, max_results: int = None):
        all_data = []
        for start in range(0, max_pages * 12, 12):
            for type_num in range(1, 9):
                if max_results is not None and len(all_data) >= max_results:
                    return all_data


                logging.info(f"Scraping page with start={start}, type={type_num}")
                html_content = self.get_page_content(start, type_num)
                if not html_content:
                    continue
                
                if max_results is not None:
                    max_limit = max_results - len(all_data)
                else:
                    max_limit = None

                page_data = self.extract_table_data(html_content, max_limit=max_limit)

                # page_data = self.extract_table_data(html_content, max_limit=max_results - len(all_data))
                all_data.extend(page_data)
                time.sleep(1)

        return all_data

    def save_to_csv(self, data: List[Dict], filename: str = 'shl_enhanced_assessments.csv'):
        if not data:
            logging.warning("No data to save")
            return

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logging.info(f"Saved {len(data)} records to {filename}")

def main():
    scraper = SHLScraper()
    logging.info("Starting SHL product catalog scraping (Enhanced)...")

    data = scraper.scrape_all_tables(max_pages=100, max_results=None)
    logging.info(f"Total records scraped: {len(data)}")

    scraper.save_to_csv(data)
    logging.info("Scraping completed!")

if __name__ == "__main__":
    main()
