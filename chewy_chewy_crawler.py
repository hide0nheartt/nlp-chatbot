import json
import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

INPUT_FILE = "json_file/chewychewy_all_urls.json"
OUTPUT_FILE = "output_files/chewychewy_products.json"

BASE_URL = "https://chewychewy.vn"

headers = {
    "User-Agent": "Mozilla/5.0"
}


def get_meta_content(soup, property_name):
    tag = soup.find("meta", attrs={"property": property_name})
    return tag["content"].strip() if tag and tag.get("content") else ""


def get_text(soup, selector):
    element = soup.select_one(selector)
    return element.get_text(" ", strip=True) if element else ""


def clean_url(url):
    return url.split("?")[0].split("#")[0].strip()


def normalize_price(price):
    digits = re.sub(r"[^\d]", "", str(price))
    return f"{digits} VND" if digits else ""


def detect_category(title, description, url):
    text = f"{title} {description} {url}".lower()

    if any(x in text for x in [
        "soda", "trà", "tra-", "cafe", "cà phê",
        "latte", "sữa tươi", "sữa dừa", "bạc sỉu",
        "hibicus", "yuzu", "boba"
    ]):
        return "Thức uống"

    if any(x in text for x in [
        "event cake", "event-cake", "banh-event",
        "bánh sinh nhật", "sinh nhật"
    ]):
        return "Bánh Event sinh nhật"

    if any(x in text for x in [
        "set mini", "mini-", "set-mini",
        "set kid", "set trio", "set "
    ]):
        return "Set bánh mini Chewy"

    if any(x in text for x in [
        "bánh su", "medium", "chewy original",
        "chewy chocolate", "cheese", "crunch",
        "black pearl", "green tea", "almond",
        "blueberry", "strawberry", "mango", "apple"
    ]):
        return "Bánh su Medium"

    return "Khác"


def collect_product_links_from_page(url):
    product_links = []

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]

            if "/products/" in href:
                full_url = urljoin(BASE_URL, href)
                product_links.append(clean_url(full_url))

    except Exception as e:
        print(f"Lỗi lấy product links từ {url}: {e}")

    return product_links


def crawl_product(url):
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    title = get_meta_content(soup, "og:title") or get_text(soup, "h1")
    description = get_meta_content(soup, "og:description")
    image_url = get_meta_content(soup, "og:image")

    price = ""

    price_meta = soup.find("meta", attrs={"property": "product:price:amount"})
    if price_meta and price_meta.get("content"):
        price = price_meta["content"]

    if not price:
        price_selectors = [
            ".price",
            ".product-price",
            ".product__price",
            "[class*=price]"
        ]

        for selector in price_selectors:
            price = get_text(soup, selector)
            if price:
                break

    product = {
        "url": clean_url(url),
        "title": title.strip(),
        "description": description.strip(),
        "price": normalize_price(price),
        "image_url": image_url.strip(),
        "category": detect_category(title, description, url)
    }

    return product


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        urls = json.load(f)

    product_urls = []
    page_urls = []

    for url in urls:
        url = clean_url(url)

        if "/products/" in url:
            product_urls.append(url)

        elif "/collections/" in url or url == BASE_URL:
            page_urls.append(url)

    print(f"Số product URL ban đầu: {len(product_urls)}")
    print(f"Số collection/homepage cần quét thêm: {len(page_urls)}")

    for page_url in page_urls:
        print("Đang lấy thêm product link từ:", page_url)
        product_urls.extend(collect_product_links_from_page(page_url))
        time.sleep(0.5)

    # Chỉ giữ product URL, bỏ rác, bỏ trùng
    product_urls = [
        url for url in product_urls
        if "/products/" in url
    ]

    product_urls = list(dict.fromkeys(product_urls))

    print(f"Tổng số product URL sạch sau khi gom: {len(product_urls)}")

    products = []

    for index, url in enumerate(product_urls, start=1):
        try:
            print(f"[{index}/{len(product_urls)}] Crawling: {url}")

            product = crawl_product(url)

            # Bỏ sản phẩm thiếu dữ liệu quan trọng
            if product["title"] and product["price"] and product["image_url"]:
                products.append(product)
            else:
                print("Bỏ qua vì thiếu title/price/image:", url)

            time.sleep(0.5)

        except Exception as e:
            print(f"Lỗi khi crawl {url}: {e}")

    os.makedirs("output_files", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=4, ensure_ascii=False)

    print(f"Đã crawl {len(products)} sản phẩm sạch.")
    print(f"Đã lưu vào: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()