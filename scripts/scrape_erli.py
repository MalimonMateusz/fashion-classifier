"""Scraper zdjec produktow modowych z erli.pl.

Zbiera miniatury z listingow kategorii (konfiguracja w scraper_config.yml) i zapisuje:
    data/01_raw/images/<klasa>/<product_id>.webp
    data/01_raw/products.csv   (product_id, label, source_category, page, title, url, image_url, image_path)

Uruchomienie:
    python scripts/scrape_erli.py                      # wszystkie klasy
    python scripts/scrape_erli.py --per-class 100      # szybki test
    python scripts/scrape_erli.py --classes zegarki bluzy_damskie

Zachowanie:
- dane sa dopisywane do products.csv po kazdej stronie listingu, wiec przerwanie
  (Ctrl+C, blokada, crash) nic nie traci,
- postep jest odtwarzany z products.csv: kazda kategoria wznawia sie od strony
  po ostatniej zapisanej,
- gdy erli wlaczy ochrone anty-botowa (403 + captcha), scraper NIE probuje jej
  obchodzic - czeka (wykladniczy backoff), a po odblokowaniu zwalnia tempo.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta
from http import HTTPStatus
from pathlib import Path

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "01_raw"
PRODUCTS_CSV = RAW_DIR / "products.csv"
PRODUCT_ID_RE = re.compile(r"/produkt/[^\"?#]*,(\d+)")
IMAGE_SIZE_RE = re.compile(r"\.(s2|l)\.webp$")
# CDN dla produktow bez zdjecia zwraca placeholder SVG - takich nie chcemy
RASTER_TYPES = ("image/webp", "image/jpeg", "image/png")
# ~1-2% produktow nie ma zdjecia, wiec kategoria jest "pelna" troche ponizej limitu
DONE_RATIO = 0.95

log = logging.getLogger("scraper")


class GiveUpError(Exception):
    """Laczny czas czekania na odblokowanie przekroczyl limit z konfiguracji."""


@dataclass
class Product:
    product_id: str
    label: str
    source_category: str
    page: int
    title: str
    url: str
    image_url: str
    image_path: str = ""


class Throttle:
    """Pilnuje tempa zapytan i reaguje na blokade anty-botowa.

    Po kazdej blokadzie: czeka (15 min -> 30 min -> 60 min ...), a po odblokowaniu
    zwieksza odstep miedzy zapytaniami, zeby nie wpasc w nia od razu ponownie.
    """

    def __init__(self, cfg: dict):
        self.delay = cfg["request_delay_s"]
        self.max_delay = cfg["max_request_delay_s"]
        self.slowdown = cfg["slowdown_factor"]
        self.initial_wait = cfg["block_wait_initial_s"]
        self.max_wait = cfg["block_wait_max_s"]
        self.give_up_after = cfg["give_up_after_s"]
        self.wait = self.initial_wait
        self.total_waited = 0.0

    def pause(self) -> None:
        time.sleep(self.delay)

    def on_block(self, url: str) -> None:
        if self.total_waited + self.wait > self.give_up_after:
            raise GiveUpError(
                f"czekano juz {self.total_waited / 3600:.1f} h, poddaje sie"
            )
        wait = self.wait * random.uniform(
            1.0, 1.2
        )  # jitter, zeby nie trafiac w rowne granice okna
        resume_at = datetime.now() + timedelta(seconds=wait)
        log.warning(
            "Blokada anty-botowa (403) na %s - czekam %.0f min (do %s)",
            url,
            wait / 60,
            resume_at.strftime("%H:%M"),
        )
        time.sleep(wait)
        self.total_waited += wait
        self.wait = min(self.wait * 2, self.max_wait)
        self.delay = min(self.delay * self.slowdown, self.max_delay)

    def on_success(self) -> None:
        self.wait = self.initial_wait


def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {"User-Agent": user_agent, "Accept-Language": "pl-PL,pl;q=0.9"}
    )
    retry = Retry(total=4, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch(
    session: requests.Session, url: str, throttle: Throttle, *, pause: bool = True
) -> requests.Response:
    """GET z obsluga blokady: przy 403 czeka i ponawia, az do skutku albo GiveUpError."""
    while True:
        resp = session.get(url, timeout=30)
        if pause:
            throttle.pause()
        if resp.status_code != HTTPStatus.FORBIDDEN:
            throttle.on_success()
            return resp
        throttle.on_block(url)


def parse_listing(html: str) -> list[dict]:
    """Wyciaga karty produktow z HTML listingu (klasy CSS sa zahashowane, wiec opieramy sie na strukturze)."""
    soup = BeautifulSoup(html, "lxml")
    items = []
    for a in soup.select('a[href*="/produkt/"]'):
        img = a.find("img")
        match = PRODUCT_ID_RE.search(a.get("href", ""))
        if img is None or match is None:
            continue
        # leniwie ladowane obrazki maja placeholder w src i prawdziwy URL w data-src
        image_url = img.get("data-src") or img.get("src") or ""
        if "i.erli.pl" not in image_url:
            continue
        title_tag = a.find("h2")
        items.append(
            {
                "product_id": match.group(1),
                "title": title_tag.get_text(strip=True)
                if title_tag
                else img.get("alt", ""),
                "url": a["href"],
                "image_url": image_url,
            }
        )
    return items


def download_image(
    session: requests.Session, product: Product, image_size: str, throttle: Throttle
) -> bool:
    path = RAW_DIR / "images" / product.label / f"{product.product_id}.webp"
    product.image_path = path.relative_to(ROOT).as_posix()
    if path.exists() and path.stat().st_size > 0:
        return True
    url = IMAGE_SIZE_RE.sub(f".{image_size}.webp", product.image_url)
    try:
        resp = fetch(session, url, throttle, pause=False)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Nie udalo sie pobrac %s: %s", url, exc)
        return False
    if not resp.headers.get("Content-Type", "").startswith(RASTER_TYPES):
        return False  # placeholder "brak zdjecia"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return True


def append_products(products: list[Product]) -> None:
    if not products:
        return
    df = pd.DataFrame(
        [asdict(p) for p in products], columns=[f.name for f in fields(Product)]
    )
    df.to_csv(PRODUCTS_CSV, mode="a", header=not PRODUCTS_CSV.exists(), index=False)


def load_progress() -> pd.DataFrame:
    if not PRODUCTS_CSV.exists():
        return pd.DataFrame(columns=[f.name for f in fields(Product)])
    return pd.read_csv(PRODUCTS_CSV, dtype={"product_id": str})


def scrape_category(  # noqa: PLR0913
    session: requests.Session,
    cfg: dict,
    throttle: Throttle,
    label: str,
    category: str,
    quota: int,
    progress: pd.DataFrame,
) -> int:
    """Scrapuje jedna kategorie do osiagniecia `quota` zdjec. Zwraca liczbe zdjec w kategorii."""
    done = progress[progress["source_category"] == category]
    count = len(done)
    seen = set(progress.loc[progress["label"] == label, "product_id"])
    # ostatnia strona mogla byc przerwana w polowie - pobieramy ja ponownie, `seen` odfiltruje duplikaty
    start_page = int(done["page"].max()) if count else 1
    if count >= DONE_RATIO * quota:
        log.info("%-28s %-50s juz gotowe (%d)", label, category, count)
        return count

    for page in range(start_page, cfg["max_pages"] + 1):
        url = f"{cfg['base_url']}/{category}" + (f"?page={page}" if page > 1 else "")
        resp = fetch(session, url, throttle)
        if resp.status_code != HTTPStatus.OK:
            log.warning(
                "%s -> HTTP %s, pomijam reszte kategorii", url, resp.status_code
            )
            break
        # po przekroczeniu ostatniej strony erli przekierowuje na pierwsza strone kategorii
        if page > 1 and "page=" not in resp.url:
            break
        items = parse_listing(resp.text)
        if not items:
            break

        page_products = []
        for item in items:
            if item["product_id"] in seen or count + len(page_products) >= quota:
                continue
            seen.add(item["product_id"])
            product = Product(label=label, source_category=category, page=page, **item)
            if download_image(session, product, cfg["image_size"], throttle):
                page_products.append(product)
            time.sleep(cfg["image_delay_s"])
        append_products(page_products)
        count += len(page_products)
        log.info(
            "%-28s %-50s str. %3d  +%2d  -> %4d/%d  (odstep %.1fs)",
            label,
            category,
            page,
            len(page_products),
            count,
            quota,
            throttle.delay,
        )
        if count >= quota:
            break
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("scraper_config.yml")
    )
    parser.add_argument(
        "--per-class", type=int, help="nadpisuje per_class z konfiguracji"
    )
    parser.add_argument("--classes", nargs="+", help="scrapuj tylko te klasy")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    per_class = args.per_class or cfg["per_class"]
    classes = {
        k: v for k, v in cfg["classes"].items() if not args.classes or k in args.classes
    }
    session = make_session(cfg["user_agent"])
    throttle = Throttle(cfg)

    try:
        for label, categories in classes.items():
            quota = per_class // len(categories)
            for category in categories:
                scrape_category(
                    session, cfg, throttle, label, category, quota, load_progress()
                )
    except GiveUpError as exc:
        log.error("%s - zebrane dane sa zapisane, uruchom ponownie pozniej", exc)
    except KeyboardInterrupt:
        log.info(
            "Przerwano - zebrane dane sa zapisane, kolejne uruchomienie wznowi prace"
        )

    df = load_progress()
    log.info(
        "products.csv: %d rekordow. Rozklad klas:\n%s",
        len(df),
        df["label"].value_counts().to_string(),
    )


if __name__ == "__main__":
    main()
