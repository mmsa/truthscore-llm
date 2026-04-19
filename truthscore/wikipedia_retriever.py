"""
Live Wikipedia retrieval via the MediaWiki Action API (no extra dependencies).

https://www.mediawiki.org/wiki/API:Search
https://www.mediawiki.org/wiki/API:Query

Set ``TRUTHSCORE_USER_AGENT`` to identify your app (Wikimedia requires a descriptive User-Agent).
"""

from __future__ import annotations

import html
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from truthscore.retrieve import pairwise_token_cosine

logger = logging.getLogger(__name__)


class WikipediaRetriever:
    """
    Retrieve short intro extracts for a claim-sized query using on-wiki search.

    This is suitable as a **broad** evidence source. Pair with ``SimilarityEvidenceVerifier``
    (default production) or any custom ``ClaimVerifier``.
    """

    def __init__(
        self,
        *,
        lang: str = "en",
        user_agent: Optional[str] = None,
        timeout_s: float = 15.0,
        api_url: Optional[str] = None,
    ) -> None:
        self._lang = (lang or "en").lower()
        self._timeout = float(timeout_s)
        self._ua = (
            user_agent
            or os.environ.get("TRUTHSCORE_USER_AGENT")
            or "TruthScore-LLM/0.2 (+https://github.com/mmsa/truthscore-llm)"
        )
        if api_url:
            self._api = api_url.rstrip("/")
        else:
            self._api = f"https://{self._lang}.wikipedia.org/w/api.php"

    def similarity(self, a: str, b: str) -> float:
        """Token cosine between two strings (for ``SimilarityEvidenceVerifier``)."""
        return pairwise_token_cosine(a, b)

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{self._api}?{q}"
        req = urllib.request.Request(url, headers={"User-Agent": self._ua})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            logger.error("Wikipedia HTTP error: %s %s", e.code, e.reason)
            raise
        except urllib.error.URLError as e:
            logger.error("Wikipedia network error: %s", e.reason)
            raise

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        search = self._request(
            {
                "action": "query",
                "list": "search",
                "srsearch": query.strip()[:300],
                "srlimit": min(max(1, top_k), 10),
                "format": "json",
            }
        )
        hits = (search.get("query") or {}).get("search") or []
        if not hits:
            return []

        pageids = [str(h["pageid"]) for h in hits if "pageid" in h][:top_k]
        if not pageids:
            return []

        extracts = self._request(
            {
                "action": "query",
                "prop": "extracts",
                "exintro": "true",
                "explaintext": "true",
                "pageids": "|".join(pageids),
                "format": "json",
            }
        )
        pages = (extracts.get("query") or {}).get("pages") or {}
        out: List[Dict[str, Any]] = []
        rank = 0
        for pid in pageids:
            page = pages.get(pid)
            if not page or page.get("missing"):
                continue
            title = page.get("title", "")
            raw = page.get("extract") or ""
            text = html.unescape(raw).strip()
            if not text:
                continue
            out.append(
                {
                    "text": text,
                    "source": f"wikipedia:{self._lang}:{title.replace(' ', '_')}",
                    "relevance": max(0.0, 1.0 - 0.07 * rank),
                    "rank": rank,
                    "title": title,
                }
            )
            rank += 1
        return out
