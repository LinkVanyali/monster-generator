"""
Fetches all posts from The Monsters Know What They're Doing (themonstersknow.com)
via the WordPress REST API and saves them as structured JSON.

Output: monsters_know_posts.json — array of post objects with plain-text content.

Usage: python fetch_monsters_know.py
"""

import json
import re
import time
import urllib.request
from html.parser import HTMLParser

BASE_URL = "https://www.themonstersknow.com/wp-json/wp/v2"
OUT_FILE = "monsters_know_posts.json"
PER_PAGE = 100  # WP API max


class HTMLStripper(HTMLParser):
    """Converts HTML to plain text, preserving paragraph breaks."""

    def __init__(self):
        super().__init__()
        self.chunks = []
        self._in_block = False

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "h1", "h2", "h3", "h4", "li", "blockquote"):
            self.chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in ("p", "h1", "h2", "h3", "h4", "li", "blockquote"):
            self.chunks.append("\n")

    def handle_data(self, data):
        self.chunks.append(data)

    def get_text(self):
        raw = "".join(self.chunks)
        # Collapse runs of blank lines to a single blank line
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def strip_html(html: str) -> str:
    stripper = HTMLStripper()
    stripper.feed(html)
    return stripper.get_text()


def fetch_json(url: str) -> tuple[list, dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "MonsterKnowArchiver/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        headers = dict(resp.headers)
        data = json.loads(resp.read().decode("utf-8"))
    return data, headers


def fetch_all_posts() -> list[dict]:
    posts = []
    page = 1

    while True:
        url = f"{BASE_URL}/posts?per_page={PER_PAGE}&page={page}&_fields=id,slug,date,title,content,categories,tags"
        print(f"  Fetching page {page} (collected {len(posts)} so far) …")

        data, _ = fetch_json(url)

        if not data:
            break

        for post in data:
            posts.append({
                "id": post["id"],
                "slug": post["slug"],
                "date": post["date"],
                "title": post["title"]["rendered"],
                "content_text": strip_html(post["content"]["rendered"]),
                "categories": post.get("categories", []),
                "tags": post.get("tags", []),
            })

        # Last page is any page that returns fewer than PER_PAGE results
        if len(data) < PER_PAGE:
            break

        page += 1
        time.sleep(0.3)  # be polite to the server

    return posts


def fetch_taxonomy(endpoint: str) -> dict[int, str]:
    """Returns id→name mapping for categories or tags."""
    mapping = {}
    page = 1
    while True:
        url = f"{BASE_URL}/{endpoint}?per_page=100&page={page}"
        data, headers = fetch_json(url)
        if not data:
            break
        for item in data:
            mapping[item["id"]] = item["name"]
        total_pages = int(headers.get("X-Wp-Totalpages", headers.get("x-wp-totalpages", 1)))
        if page >= total_pages:
            break
        page += 1
    return mapping


def main():
    print("Fetching categories and tags …")
    categories = fetch_taxonomy("categories")
    tags = fetch_taxonomy("tags")

    print("Fetching posts …")
    posts = fetch_all_posts()

    # Resolve IDs to human-readable names
    for post in posts:
        post["category_names"] = [categories.get(c, str(c)) for c in post["categories"]]
        post["tag_names"] = [tags.get(t, str(t)) for t in post["tags"]]

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(posts)} posts saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
