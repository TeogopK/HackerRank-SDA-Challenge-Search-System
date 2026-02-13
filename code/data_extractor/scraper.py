from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
import os
import json
import time
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys
import re
from html import unescape

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / f'scraper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ==================== AUTHENTICATION ====================
def authenticate():
    """Login to HackerRank and save authentication state"""
    load_dotenv(PROJECT_ROOT / ".env")
    EMAIL = os.getenv("HACKERRANK_EMAIL")
    PASSWORD = os.getenv("HACKERRANK_PASSWORD")

    if not EMAIL or not PASSWORD:
        raise ValueError("HACKERRANK_EMAIL and HACKERRANK_PASSWORD must be set in .env file")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        print("Navigating to login page...")
        page.goto("https://www.hackerrank.com/auth/login")
        page.wait_for_selector('input[name="username"]', timeout=10000)

        print("Filling credentials...")
        page.fill('input[name="username"]', EMAIL)
        page.fill('input[name="password"]', PASSWORD)

        print("Submitting login...")
        page.click('button[type="submit"]')

        try:
            page.wait_for_url("**/dashboard**", timeout=60000)
        except:
            page.wait_for_load_state("networkidle", timeout=60000)

        time.sleep(2)

        print("Saving authentication state...")
        context.storage_state(path=str(PROJECT_ROOT / "auth.json"))
        browser.close()

    print("[OK] Logged in and saved auth.json")


# ==================== HELPER FUNCTIONS ====================
def get_csrf_token(page):
    """Extract CSRF token from page"""
    token = page.evaluate(
        """() => {
            const meta = document.querySelector('meta[name="csrf-token"]');
            if (meta) return meta.getAttribute('content');
            return null;
        }"""
    )
    if token:
        return token

    cookies = page.context.cookies()
    for cookie in cookies:
        if "csrf" in cookie["name"].lower() or "xsrf" in cookie["name"].lower():
            return cookie["value"]

    token = page.evaluate(
        """() => {
            if (window._csrf) return window._csrf;
            if (window.csrfToken) return window.csrfToken;
            try {
                const ls = localStorage.getItem('csrfToken');
                if (ls) return ls;
            } catch(e) {}
            return null;
        }"""
    )
    if token:
        return token

    html = page.content()
    match = re.search(r'csrf["\']?\s*[:=]\s*["\']([^"\']+)["\']', html, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def clean_html(html_text):
    """Remove HTML tags and decode special characters"""
    if not html_text:
        return ""

    text = re.sub(r"<style[^>]*>.*?</style>", "", html_text, flags=re.DOTALL)
    text = re.sub(r"<svg[^>]*>.*?</svg>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = text.strip()

    return text


def extract_contest_slug_from_url(url):
    """Extract contest slug from HackerRank URL"""
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    url = re.sub(r"^hackerrank\.com/", "", url)
    url = re.sub(r"^contests/", "", url)
    url = re.split(r"[/?#]", url)[0]
    return url.strip()


def load_contests_from_csv(csv_path, url_column="link"):
    """Load contest slugs from CSV file"""
    try:
        df = pd.read_csv(csv_path)

        if url_column not in df.columns:
            logger.error(f"Column '{url_column}' not found. Available: {df.columns.tolist()}")
            return []

        contest_slugs = df[url_column].apply(extract_contest_slug_from_url).tolist()

        seen = set()
        unique_slugs = []
        for slug in contest_slugs:
            if slug and slug not in seen:
                seen.add(slug)
                unique_slugs.append(slug)

        logger.info(f"Loaded {len(unique_slugs)} unique contests from {csv_path}")
        return unique_slugs

    except Exception as e:
        logger.error(f"Error loading contests from CSV: {e}")
        return []


# ==================== SCRAPING FUNCTIONS ====================
def fetch_contest_challenges(page, contest_slug):
    """Fetch list of challenges in a contest"""
    url = f"https://www.hackerrank.com/contests/{contest_slug}/challenges"
    logger.info(f"  Navigating to: {url}")

    try:
        page.goto(url, wait_until="load", timeout=30000)
    except Exception as e:
        logger.warning(f"  Navigation timeout: {e}")

    time.sleep(3)
    csrf = get_csrf_token(page)

    if not csrf:
        logger.warning("  No CSRF token found, trying alternative approach...")
        data = page.evaluate(
            """() => {
                if (window.__initialData) return window.__initialData;
                if (window.__INITIAL_STATE__) return window.__INITIAL_STATE__;
                return null;
            }"""
        )
        if data:
            logger.info("  [OK] Found data in window object")
            return data
        csrf = ""

    headers = {
        "Accept": "application/json",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": url,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    if csrf:
        headers["X-CSRF-Token"] = csrf

    api_url = f"https://www.hackerrank.com/rest/contests/{contest_slug}/challenges"
    logger.info(f"  Fetching challenges list: {api_url}")

    response = page.request.get(api_url, headers=headers)

    if not response.ok:
        logger.error(f"  Response status: {response.status}")
        raise RuntimeError(f"{contest_slug} -> HTTP {response.status}")

    return response.json()


def fetch_challenge_detail(page, contest_slug, challenge_slug):
    """Fetch detailed information for a specific challenge"""
    url = f"https://www.hackerrank.com/contests/{contest_slug}/challenges/{challenge_slug}"

    try:
        page.goto(url, wait_until="load", timeout=30000)
    except Exception as e:
        logger.warning(f"    Navigation timeout for {challenge_slug}: {e}")

    time.sleep(2)
    csrf = get_csrf_token(page)

    challenge_data = page.evaluate(
        """() => {
            if (window.model) return window.model;
            if (window.__initialData && window.__initialData.model) return window.__initialData.model;
            return null;
        }"""
    )

    if challenge_data:
        return challenge_data

    if not csrf:
        csrf = ""

    headers = {
        "Accept": "application/json",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": url,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    if csrf:
        headers["X-CSRF-Token"] = csrf

    api_url = f"https://www.hackerrank.com/rest/contests/{contest_slug}/challenges/{challenge_slug}"
    response = page.request.get(api_url, headers=headers)

    if not response.ok:
        logger.error(f"    Failed to fetch {challenge_slug}: HTTP {response.status}")
        return None

    return response.json()


# ==================== DATA STORAGE ====================
class DataManager:
    """Manages all data storage and deduplication"""

    def __init__(self, base_dir=str(PROJECT_ROOT / "scraped_data")):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        self.contests_dir = self.base_dir / "contests"
        self.challenges_dir = self.base_dir / "challenges"
        self.mappings_dir = self.base_dir / "mappings"

        for d in [self.contests_dir, self.challenges_dir, self.mappings_dir]:
            d.mkdir(exist_ok=True)

        self.contests_csv = self.base_dir / "all_contests.csv"
        self.challenges_csv = self.base_dir / "all_challenges_unique.csv"
        self.contest_challenge_map_csv = self.base_dir / "contest_challenge_mapping.csv"

        self.contests_df = self._load_or_create_df(self.contests_csv)
        self.challenges_df = self._load_or_create_df(self.challenges_csv)
        self.mapping_df = self._load_or_create_df(self.contest_challenge_map_csv)

        logger.info(f"Loaded {len(self.contests_df)} contests, {len(self.challenges_df)} unique challenges")

    def _load_or_create_df(self, path):
        """Load existing CSV or create empty DataFrame"""
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}, creating new")
                return pd.DataFrame()
        return pd.DataFrame()

    def save_contest_metadata(self, contest_slug, contest_data):
        """Save contest metadata"""
        try:
            json_path = self.contests_dir / f"{contest_slug}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(contest_data, f, ensure_ascii=False, indent=2)

            metadata = {
                "contest_slug": contest_slug,
                "total_challenges": len(contest_data.get("models", [])),
                "scraped_at": datetime.now().isoformat(),
            }

            if self.contests_df.empty or "contest_slug" not in self.contests_df.columns:
                self.contests_df = pd.DataFrame([metadata])
            elif contest_slug in self.contests_df["contest_slug"].values:
                self.contests_df.loc[self.contests_df["contest_slug"] == contest_slug, "total_challenges"] = metadata[
                    "total_challenges"
                ]
                self.contests_df.loc[self.contests_df["contest_slug"] == contest_slug, "scraped_at"] = metadata[
                    "scraped_at"
                ]
            else:
                self.contests_df = pd.concat([self.contests_df, pd.DataFrame([metadata])], ignore_index=True)

            self.contests_df.to_csv(self.contests_csv, index=False)
            logger.info(f"  [OK] Saved contest metadata: {contest_slug}")

        except Exception as e:
            logger.error(f"  [ERROR] Error saving contest metadata: {e}")

    def save_challenge(self, contest_slug, challenge_data):
        """Save challenge data with deduplication"""
        try:
            if isinstance(challenge_data, dict) and "model" in challenge_data:
                challenge = challenge_data["model"]
            else:
                challenge = challenge_data

            challenge_slug = challenge.get("slug")
            if not challenge_slug:
                logger.warning("  Challenge has no slug, skipping")
                return False

            json_path = self.challenges_dir / f"{challenge_slug}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(challenge, f, ensure_ascii=False, indent=2)

            self._save_challenge_text(challenge_slug, challenge)

            challenge_row = {
                "challenge_slug": challenge_slug,
                "name": challenge.get("name", ""),
                "difficulty": challenge.get("difficulty_name", ""),
                "max_score": challenge.get("max_score", 0),
                "success_ratio": challenge.get("success_ratio", 0),
                "total_count": challenge.get("total_count", 0),
                "solved_count": challenge.get("solved_count", 0),
                "category": challenge.get("category", ""),
                "languages_count": len(challenge.get("languages", [])),
                "scraped_at": datetime.now().isoformat(),
            }

            if self.challenges_df.empty or "challenge_slug" not in self.challenges_df.columns:
                self.challenges_df = pd.DataFrame([challenge_row])
                logger.info(f"    [NEW] {challenge_slug}")
            elif challenge_slug in self.challenges_df["challenge_slug"].values:
                for key, value in challenge_row.items():
                    if key != "challenge_slug":
                        self.challenges_df.loc[self.challenges_df["challenge_slug"] == challenge_slug, key] = value
                logger.info(f"    [UPDATE] {challenge_slug}")
            else:
                self.challenges_df = pd.concat([self.challenges_df, pd.DataFrame([challenge_row])], ignore_index=True)
                logger.info(f"    [NEW] {challenge_slug}")

            mapping_row = {"contest_slug": contest_slug, "challenge_slug": challenge_slug}

            if self.mapping_df.empty or "contest_slug" not in self.mapping_df.columns:
                self.mapping_df = pd.DataFrame([mapping_row])
            else:
                mask = (self.mapping_df["contest_slug"] == contest_slug) & (
                    self.mapping_df["challenge_slug"] == challenge_slug
                )
                if not mask.any():
                    self.mapping_df = pd.concat([self.mapping_df, pd.DataFrame([mapping_row])], ignore_index=True)

            return True

        except Exception as e:
            logger.error(f"  [ERROR] Error saving challenge: {e}")
            return False

    def _save_challenge_text(self, challenge_slug, challenge):
        """Save readable text version of challenge"""
        try:
            txt_path = self.challenges_dir / f"{challenge_slug}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Challenge: {challenge.get('name', challenge_slug)}\n")
                f.write(f"Slug: {challenge.get('slug', challenge_slug)}\n")
                f.write(f"Difficulty: {challenge.get('difficulty_name', 'N/A')}\n")
                f.write(f"Max Score: {challenge.get('max_score', 'N/A')}\n")
                f.write(f"Success Ratio: {challenge.get('success_ratio', 'N/A')}\n")
                f.write(f"Solved: {challenge.get('solved_count', 0)}/{challenge.get('total_count', 0)}\n")
                f.write("\n" + "=" * 70 + "\n\n")

                if "problem_statement" in challenge:
                    f.write("PROBLEM STATEMENT:\n")
                    f.write(clean_html(challenge["problem_statement"]))
                    f.write("\n\n")

                if "input_format" in challenge:
                    f.write("INPUT FORMAT:\n")
                    f.write(clean_html(challenge["input_format"]))
                    f.write("\n\n")

                if "constraints" in challenge:
                    f.write("CONSTRAINTS:\n")
                    f.write(clean_html(challenge["constraints"]))
                    f.write("\n\n")

                if "output_format" in challenge:
                    f.write("OUTPUT FORMAT:\n")
                    f.write(clean_html(challenge["output_format"]))
                    f.write("\n\n")
        except Exception as e:
            logger.error(f"Error saving text for {challenge_slug}: {e}")

    def save_all(self):
        """Save all DataFrames to CSV"""
        try:
            if not self.contests_df.empty:
                self.contests_df.to_csv(self.contests_csv, index=False)
            if not self.challenges_df.empty:
                self.challenges_df.to_csv(self.challenges_csv, index=False)
            if not self.mapping_df.empty:
                self.mapping_df.to_csv(self.contest_challenge_map_csv, index=False)
            logger.info("[OK] Saved all master datasets")
        except Exception as e:
            logger.error(f"[ERROR] Error saving datasets: {e}")

    def get_stats(self):
        """Get current statistics"""
        return {
            "total_contests": len(self.contests_df),
            "total_unique_challenges": len(self.challenges_df),
            "total_mappings": len(self.mapping_df),
        }


# ==================== MAIN SCRAPING ====================
def scrape_contests(
    contest_slugs,
    scrape_details=True,
    headless=True,
    delay_between_contests=3,
    delay_between_challenges=1,
    skip_existing_contests=True,
    skip_existing_challenges=True,
    max_workers=3,
):
    """
    Scrape multiple contests with error handling and concurrency

    Args:
        contest_slugs: List of contest slugs to scrape
        scrape_details: Whether to fetch full challenge details
        headless: Run browser in headless mode
        delay_between_contests: Seconds to wait between contests
        delay_between_challenges: Seconds to wait between challenges
        skip_existing_contests: Skip contests that are already scraped
        skip_existing_challenges: Skip challenges that are already scraped
        max_workers: Number of parallel browser contexts (1-5 recommended, be nice to servers!)
    """
    if not (PROJECT_ROOT / "auth.json").exists():
        logger.error("auth.json not found. Please run authenticate() first.")
        return

    data_manager = DataManager()

    # Filter out already scraped contests if requested
    if (
        skip_existing_contests
        and not data_manager.contests_df.empty
        and "contest_slug" in data_manager.contests_df.columns
    ):
        existing_contests = set(data_manager.contests_df["contest_slug"].tolist())
        original_count = len(contest_slugs)
        contest_slugs = [slug for slug in contest_slugs if slug not in existing_contests]
        skipped = original_count - len(contest_slugs)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already scraped contests, {len(contest_slugs)} remaining")

    if not contest_slugs:
        logger.info("No new contests to scrape!")
        return

    successful = 0
    failed = 0
    failed_contests = []

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=headless)

            # Use single context for simplicity and rate limiting
            if max_workers == 1:
                context = browser.new_context(storage_state=str(PROJECT_ROOT / "auth.json"))
                context.set_extra_http_headers(
                    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                )
                page = context.new_page()

                for idx, slug in enumerate(contest_slugs, 1):
                    result = scrape_single_contest(
                        page,
                        slug,
                        idx,
                        len(contest_slugs),
                        data_manager,
                        scrape_details,
                        delay_between_challenges,
                        skip_existing_challenges,
                    )

                    if result["success"]:
                        successful += 1
                    else:
                        failed += 1
                        failed_contests.append(slug)

                    if idx < len(contest_slugs):
                        logger.info(f"Waiting {delay_between_contests}s before next contest...")
                        time.sleep(delay_between_contests)

                context.close()

            else:
                # Multi-context scraping (parallel)
                from concurrent.futures import ThreadPoolExecutor, as_completed

                def scrape_contest_worker(contest_info):
                    """Worker function for parallel scraping"""
                    slug, idx = contest_info
                    context = browser.new_context(storage_state=str(PROJECT_ROOT / "auth.json"))
                    context.set_extra_http_headers(
                        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                    )
                    page = context.new_page()

                    result = scrape_single_contest(
                        page,
                        slug,
                        idx,
                        len(contest_slugs),
                        data_manager,
                        scrape_details,
                        delay_between_challenges,
                        skip_existing_challenges,
                    )

                    context.close()
                    time.sleep(delay_between_contests)  # Rate limiting
                    return result

                # Process in parallel
                contest_info_list = [(slug, idx) for idx, slug in enumerate(contest_slugs, 1)]

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(scrape_contest_worker, info): info for info in contest_info_list}

                    for future in as_completed(futures):
                        result = future.result()
                        if result["success"]:
                            successful += 1
                        else:
                            failed += 1
                            failed_contests.append(result["slug"])

            browser.close()

        except Exception as e:
            logger.error(f"Fatal error: {e}")

        finally:
            data_manager.save_all()

            stats = data_manager.get_stats()
            logger.info(f"\n{'='*70}")
            logger.info("SCRAPING SUMMARY")
            logger.info("=" * 70)
            logger.info(f"Contests: {successful} successful, {failed} failed")
            logger.info(f"Total unique challenges: {stats['total_unique_challenges']}")
            logger.info(f"Total mappings: {stats['total_mappings']}")

            if failed_contests:
                logger.info(f"\nFailed contests: {', '.join(failed_contests)}")

            logger.info(f"\nData saved to: {data_manager.base_dir.absolute()}")


def scrape_single_contest(
    page, slug, idx, total, data_manager, scrape_details, delay_between_challenges, skip_existing_challenges
):
    """Scrape a single contest"""
    logger.info(f"\n{'='*70}")
    logger.info(f"[{idx}/{total}] Scraping contest: {slug}")
    logger.info("=" * 70)

    try:
        contest_data = fetch_contest_challenges(page, slug)
        data_manager.save_contest_metadata(slug, contest_data)

        challenges = contest_data.get("models", [])
        logger.info(f"  Found {len(challenges)} challenges")

        if scrape_details and challenges:
            # Filter out already scraped challenges if requested
            if (
                skip_existing_challenges
                and not data_manager.challenges_df.empty
                and "challenge_slug" in data_manager.challenges_df.columns
            ):
                existing_challenges = set(data_manager.challenges_df["challenge_slug"].tolist())
                original_count = len(challenges)
                challenges = [ch for ch in challenges if ch.get("slug") not in existing_challenges]
                skipped = original_count - len(challenges)
                if skipped > 0:
                    logger.info(f"  Skipping {skipped} already scraped challenges, {len(challenges)} new")

            if challenges:
                logger.info(f"  Scraping {len(challenges)} challenge details...")

                challenge_success = 0
                challenge_failed = 0

                for i, challenge in enumerate(challenges, 1):
                    challenge_slug = challenge.get("slug")
                    challenge_name = challenge.get("name", challenge_slug)

                    if not challenge_slug:
                        logger.warning(f"    [{i}/{len(challenges)}] Skipping challenge with no slug")
                        continue

                    logger.info(f"    [{i}/{len(challenges)}] {challenge_name}")

                    try:
                        detail = fetch_challenge_detail(page, slug, challenge_slug)

                        if detail:
                            if isinstance(detail, dict) and "model" in detail:
                                merged = {**challenge, **detail["model"]}
                            else:
                                merged = {**challenge, **detail}

                            if data_manager.save_challenge(slug, merged):
                                challenge_success += 1
                            else:
                                challenge_failed += 1
                        else:
                            if data_manager.save_challenge(slug, challenge):
                                challenge_success += 1
                            else:
                                challenge_failed += 1

                    except Exception as e:
                        logger.error(f"    [FAILED] Error: {e}")
                        challenge_failed += 1
                        try:
                            data_manager.save_challenge(slug, challenge)
                        except:
                            pass

                    time.sleep(delay_between_challenges)

                logger.info(f"  Challenge results: {challenge_success} success, {challenge_failed} failed")

        data_manager.save_all()
        logger.info(f"[SUCCESS] Successfully scraped contest: {slug}")
        return {"success": True, "slug": slug}

    except Exception as e:
        logger.error(f"[FAILED] Error scraping contest {slug}: {e}")
        return {"success": False, "slug": slug}


# ==================== MAIN ====================
if __name__ == "__main__":
    # Step 1: Authenticate (run once)
    authenticate()

    # Step 2: Load contest slugs from CSV
    contest_slugs = load_contests_from_csv(csv_path=str(PROJECT_ROOT / "all_links.csv"), url_column="link")

    logger.info(f"\nTotal contests in CSV: {len(contest_slugs)}")
    logger.info(f"First 5 contests: {contest_slugs[:5]}")

    # For testing - comment out to scrape all
    # contest_slugs = contest_slugs[:3]

    if contest_slugs:
        scrape_contests(
            contest_slugs=contest_slugs,
            scrape_details=True,  # Set to False to only get contest metadata (faster)
            headless=True,  # Set to False to see browser
            delay_between_contests=2,  # Seconds between contests (be nice!)
            delay_between_challenges=0.5,  # Seconds between challenges
            skip_existing_contests=True,  # Skip already scraped contests (for re-runs)
            skip_existing_challenges=True,  # Skip already scraped challenges (for re-runs)
            max_workers=1,  # 1-5 parallel contexts (1=sequential, 3=good balance)
        )
    else:
        logger.error("No contests to scrape!")
