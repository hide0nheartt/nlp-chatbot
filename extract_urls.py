import requests
import xml.etree.ElementTree as ET
import json
import os
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin

SITEMAP_URLS = [
    "https://chewychewy.vn/sitemap_products_1.xml",
    "https://chewychewy.vn/sitemap_collections_1.xml",
]

OUTPUT_FILE = "json_file/chewychewy_all_urls.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

all_urls = []


def fetch_sitemap(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        for url_element in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
            all_urls.append(url_element.text.strip())

    except Exception as e:
        print(f"Lỗi khi lấy hoặc parse sitemap {url}: {e}")


def collect_product_links_from_page(url):
    product_links = []

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]

            if "/products/" in href:
                full_url = urljoin("https://chewychewy.vn", href)
                clean_url = full_url.split("?")[0].split("#")[0]
                product_links.append(clean_url)

    except Exception as e:
        print(f"Lỗi khi lấy product links từ {url}: {e}")

    return product_links


def main():
    # 1. Lấy URL từ sitemap
    for sitemap_url in SITEMAP_URLS:
        fetch_sitemap(sitemap_url)

    # 2. Lấy các trang collection
    collection_urls = [
        url for url in all_urls
        if "/collections/" in url or url == "https://chewychewy.vn"
    ]

    print(f"Số URL lấy từ sitemap: {len(all_urls)}")
    print(f"Số collection/homepage sẽ quét thêm: {len(collection_urls)}")

    # 3. Vào collection/homepage để lấy thêm product URL
    for collection_url in collection_urls:
        print("Đang quét thêm:", collection_url)
        product_links = collect_product_links_from_page(collection_url)
        all_urls.extend(product_links)
        time.sleep(0.5)

    # 4. Loại trùng URL
    unique_urls = list(dict.fromkeys(all_urls))

    os.makedirs("json_file", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unique_urls, f, indent=4, ensure_ascii=False)

    print(f"Extracted {len(unique_urls)} unique URLs and saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()