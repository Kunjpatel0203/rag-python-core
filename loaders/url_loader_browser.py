# import os
# import requests
# from urllib.parse import urljoin
# from playwright.sync_api import sync_playwright, TimeoutError

# SUPPORTED_EXTENSIONS = (
#     ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".csv"
# )


# def _direct_download(url, download_dir):
#     """
#     Fallback HTTP download (works for preview/new-tab documents)
#     """
#     try:
#         resp = requests.get(url, timeout=30, stream=True)
#         resp.raise_for_status()

#         filename = url.split("/")[-1].split("?")[0]
#         path = os.path.join(download_dir, filename)

#         with open(path, "wb") as f:
#             for chunk in resp.iter_content(chunk_size=8192):
#                 f.write(chunk)

#         print(f"⬇ Downloaded (direct): {filename}")
#         return path

#     except Exception as e:
#         print(f"⚠ Direct download failed: {e}")
#         return None


# def load_url_with_browser(url, download_dir="data/downloads"):
#     """
#     UNIVERSAL URL LOADER

#     Handles:
#     - Direct downloads
#     - New-tab previews
#     - JS-triggered links
#     - Gov websites
#     """

#     os.makedirs(download_dir, exist_ok=True)

#     page_text = ""
#     downloaded_files = []

#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=True)
#         context = browser.new_context(
#             accept_downloads=True,
#             ignore_https_errors=True
#         )
#         page = context.new_page()

#         # -----------------------------
#         # Load page (soft load)
#         # -----------------------------
#         try:
#             page.goto(url, timeout=30000)
#         except TimeoutError:
#             print("⚠ Page load timeout — continuing")

#         # Best-effort text extraction
#         try:
#             page_text = page.evaluate("document.body.innerText")
#         except Exception:
#             page_text = ""

#         # -----------------------------
#         # Extract all anchor links
#         # -----------------------------
#         anchors = page.query_selector_all("a[href]")

#         for anchor in anchors:
#             href = anchor.get_attribute("href")
#             if not href:
#                 continue

#             full_url = urljoin(url, href)
#             lower = full_url.lower()

#             if not lower.endswith(SUPPORTED_EXTENSIONS):
#                 continue

#             # -----------------------------
#             # 1️⃣ TRY BROWSER DOWNLOAD
#             # -----------------------------
#             try:
#                 with page.expect_download(timeout=5000) as d_info:
#                     anchor.click()

#                 download = d_info.value
#                 filename = download.suggested_filename
#                 save_path = os.path.join(download_dir, filename)

#                 download.save_as(save_path)
#                 downloaded_files.append(save_path)

#                 print(f"⬇ Downloaded (browser): {filename}")
#                 continue

#             except Exception:
#                 pass  # browser download did not happen

#             # -----------------------------
#             # 2️⃣ FALLBACK: DIRECT HTTP
#             # -----------------------------
#             path = _direct_download(full_url, download_dir)
#             if path:
#                 downloaded_files.append(path)
#             else:
#                 print(f"⚠ Skipped (protected/invalid): {full_url}")

#         browser.close()

#     return page_text, downloaded_files





import os
import requests
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright, TimeoutError

SUPPORTED_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".ppt", ".pptx",
    ".xls", ".xlsx", ".csv"
)


def _direct_download(url, download_dir):
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()

        filename = url.split("/")[-1].split("?")[0]
        path = os.path.join(download_dir, filename)

        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"⬇ Downloaded (direct): {filename}")
        return path

    except Exception as e:
        print(f"⚠ Direct download failed: {e}")
        return None


def load_url_with_browser(url, download_dir="data/downloads"):
    os.makedirs(download_dir, exist_ok=True)

    page_text = ""
    downloaded_files = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            accept_downloads=True,
            ignore_https_errors=True
        )
        page = context.new_page()

        try:
            page.goto(url, timeout=30000)
        except TimeoutError:
            print("⚠ Page load timeout — continuing")

        # Extract visible page text
        try:
            page_text = page.evaluate("document.body.innerText")
        except:
            page_text = ""

        # =====================================================
        # ✅ ONLY VISIBLE ANCHOR LINKS ARE COLLECTED
        # =====================================================
        table = page.query_selector("table")
        anchors = []

        if table:
            anchors = table.query_selector_all("a[href]")


        for anchor in anchors:
            href = anchor.get_attribute("href")
            if not href:
                continue

            full_url = urljoin(url, href)
            lower = full_url.lower()

            if not lower.endswith(SUPPORTED_EXTENSIONS):
                continue

            # 1️⃣ Try browser-triggered download
            try:
                with page.expect_download(timeout=5000) as d_info:
                    anchor.click()

                download = d_info.value
                filename = download.suggested_filename
                save_path = os.path.join(download_dir, filename)
                download.save_as(save_path)

                print(f"⬇ Downloaded (browser): {filename}")
                downloaded_files.append(save_path)
                continue

            except:
                pass

            # 2️⃣ Fallback direct HTTP download
            path = _direct_download(full_url, download_dir)
            if path:
                downloaded_files.append(path)
            else:
                print(f"⚠ Skipped: {full_url}")

        browser.close()

    return page_text, downloaded_files
