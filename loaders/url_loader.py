import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

SUPPORTED_EXT = (".pdf", ".docx", ".xlsx", ".pptx")

def load_url(url: str):
    response = requests.get(url, timeout=15)
    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    page_text = soup.get_text(separator=" ")

    doc_links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"])
        if href.lower().endswith(SUPPORTED_EXT):
            doc_links.append(href)

    return page_text.strip(), list(set(doc_links))
