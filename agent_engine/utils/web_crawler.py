# 把所有import语句放在文件顶部
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from urllib.parse import quote
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import random

class BaseWebCrawler:
    """Base web crawler class."""
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined})
            '''
        })
    
    def cleanup(self):
        """Clean up the resources."""
        self.driver.quit()

class BaikeWebCrawler(BaseWebCrawler):
    """Structured web crawler for Baidu Baike and Wikipedia.
    
    Args:
        storage_dir (str): the directory to store the cache files, default to "storage"
        storage_update_interval (int): the interval in days to update the cache, default to 30
    """
    def __init__(
            self,
            storage_dir: str = "storage",
            storage_update_interval: int = 30,
            secure_sleep_time: int = 2,
            sleep_time_variation: int = 1
            ):
        super().__init__()
        
        self.storage_dir = storage_dir
        self.storage_update_interval = storage_update_interval
        self.baidu_storage_dir = os.path.join(self.storage_dir, "baidu")
        self.wiki_storage_dir = os.path.join(self.storage_dir, "wiki")
        os.makedirs(self.baidu_storage_dir, exist_ok=True)
        os.makedirs(self.wiki_storage_dir, exist_ok=True)
        
        self.secure_sleep_time = secure_sleep_time + random.uniform(0, sleep_time_variation)
    
    def _get_cache_path(self, source: str, keyword: str) -> str:
        """Get the cache file path for the given keyword and source."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{keyword.replace(' ', '_')}-{timestamp}.html"
        if source == "baidu":
            return os.path.join(self.baidu_storage_dir, filename)
        elif source == "wiki":
            return os.path.join(self.wiki_storage_dir, filename)
        raise ValueError(f"[CRAWLER] Unknown source: {source}")
    
    def _find_valid_cache(self, source: str, keyword: str) -> str:
        """Find a valid cache file for the given keyword and source."""
        storage_dir = self.baidu_storage_dir if source == "baidu" else self.wiki_storage_dir
        pattern = f"{keyword.replace(' ', '_')}-*.html"
        for filename in os.listdir(storage_dir):
            if filename.startswith(f"{keyword.replace(' ', '_')}-"):
                parts = filename.split('-', 1)
                if len(parts) < 2:
                    continue
                timestamp_str = parts[1].split('.')[0]
                try:
                    file_time = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                    if datetime.now() - file_time < timedelta(days=self.storage_update_interval):
                        return os.path.join(storage_dir, filename)
                except ValueError:
                    continue
        return None
    
    def _save_html(self, source: str, keyword: str, html: str):
        """Save the HTML content to cache."""
        filepath = self._get_cache_path(source, keyword)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html)
        except Exception as e:
            print(f"[CRAWLER] Save storage failed: {e}")

    def _load_html(self, filepath: str) -> str:
        """Load the HTML content from cache."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[CRAWLER] Fail to load storage: {e}")
            return None

    def get_baidu_baike_content(self, keyword: str) -> str:
        """Get the content from Baidu Baike."""
        cache_file = self._find_valid_cache("baidu", keyword)
        if cache_file:
            print("[CRAWLER] Using baidu storage:", cache_file)
            html = self._load_html(cache_file)
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                paragraphs = soup.find_all('div', class_=lambda x: x and 'para' in x)
                
                if not paragraphs:
                    return f"[CRAWLER] Baidu content not found. Keyword {keyword}"
                
                content = "\n".join([p.get_text(strip=True) for p in paragraphs])
                return content

        keyword_encoded = quote(keyword)
        url = f"https://baike.baidu.com/item/{keyword_encoded}"
        
        self.driver.get(url)
        time.sleep(self.secure_sleep_time)
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        paragraphs = soup.find_all('div', class_=lambda x: x and 'para' in x)
        
        if not paragraphs:
            return f"[CRAWLER] Keyword '{keyword}' not found"
        
        content = "\n".join([p.get_text(strip=True) for p in paragraphs])
        
        self._save_html("baidu", keyword, self.driver.page_source)
        
        return content

    def get_wikipedia_content(self, keyword: str) -> str:
        """改进的维基百科内容获取方法，使用搜索功能"""
        cache_file = self._find_valid_cache("wiki", keyword)
        if cache_file:
            print("[CRAWLER] Using wiki storage:", cache_file)
            html = self._load_html(cache_file)
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                content = soup.find('div', {'id': 'mw-content-text'})
                if content:
                    return content.get_text(separator='\n', strip=True)
                return f"[CRAWLER] Wiki content not found. Keyword {keyword}"

        # 构造搜索URL
        search_url = f"https://en.wikipedia.org/w/index.php?search={quote(keyword)}&title=Special%3ASearch&ns0=1"
        self.driver.get(search_url)
        time.sleep(self.secure_sleep_time + random.uniform(0, 1))
        
        # 检查是否直接跳转到目标页面
        if "Special:Search" not in self.driver.current_url:
            # 直接获取内容
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            content = soup.find('div', {'id': 'mw-content-text'})
            if content:
                self._save_html("wiki", keyword, self.driver.page_source)
                return content.get_text(separator='\n', strip=True)
        else:
            # 解析搜索结果页面
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            search_results = soup.select(".mw-search-results .mw-search-result-heading a")
            
            if search_results:
                # 获取第一个结果的相对路径
                first_result_url = search_results[0]['href']
                full_url = f"https://en.wikipedia.org{first_result_url}"
                
                # 访问第一个结果页面
                self.driver.get(full_url)
                time.sleep(self.secure_sleep_time + random.uniform(0, 1))
                
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                content = soup.find('div', {'id': 'mw-content-text'})
                
                if content:
                    self._save_html("wiki", keyword, self.driver.page_source)
                    return content.get_text(separator='\n', strip=True)
            else:
                return f"[CRAWLER] No Wikipedia results found for: {keyword}"
        
        return f"[CRAWLER] Failed to retrieve Wikipedia content for: {keyword}"
    
    def cleanup(self):
        """Clean up the resources."""
        super().cleanup()