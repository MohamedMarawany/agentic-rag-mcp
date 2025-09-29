
# AI Agent for web search (placeholder)
import logging
from typing import Any
import requests

logger = logging.getLogger(__name__)

class WebSearchAgent:
    """
    AI Agent for performing web search queries.
    Extend with real web search logic as needed.
    """
    def search(self, query: str) -> Any:
        """
        Perform a web search for the given query using DuckDuckGo Instant Answer API, with Wikipedia and Serper fallback.
        Returns (answer, source) tuple.
        """
        import time
        try:
            logger.info(f"Performing web search for: {query}")
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }
            for attempt in range(3):
                resp = requests.get(url, params=params, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    for key in ["AbstractText", "Answer", "Definition"]:
                        val = data.get(key)
                        if val:
                            return val, "WebSearchAgent - DuckDuckGo"
                    related = data.get("RelatedTopics")
                    if related and isinstance(related, list) and len(related) > 0:
                        first = related[0]
                        if isinstance(first, dict) and first.get("Text"):
                            return first["Text"], "WebSearchAgent - DuckDuckGo"
                    break
                elif resp.status_code == 202:
                    logger.warning(f"DuckDuckGo API returned 202, retrying ({attempt+1}/3)...")
                    time.sleep(1.5)
                else:
                    logger.warning(f"DuckDuckGo API returned status {resp.status_code}, retrying ({attempt+1}/3)...")
                    time.sleep(1.5)
            logger.info("Falling back to Wikipedia summary API.")
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
            wiki_resp = requests.get(wiki_url, timeout=8)
            if wiki_resp.status_code == 200:
                wiki_data = wiki_resp.json()
                extract = wiki_data.get("extract")
                if extract:
                    return extract, "WebSearchAgent - Wikipedia"
            logger.info("Falling back to Wikipedia search API.")
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'format': 'json',
                'utf8': 1
            }
            search_resp = requests.get(search_url, params=search_params, timeout=8)
            if search_resp.status_code == 200:
                search_data = search_resp.json()
                search_results = search_data.get('query', {}).get('search', [])
                if search_results:
                    top_title = search_results[0]['title']
                    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{top_title.replace(' ', '_')}"
                    summary_resp = requests.get(summary_url, timeout=8)
                    if summary_resp.status_code == 200:
                        summary_data = summary_resp.json()
                        extract = summary_data.get("extract")
                        if extract:
                            return extract, "WebSearchAgent - Wikipedia"
            logger.info("Falling back to Serper API.")
            serper_url = "https://google.serper.dev/search"
            serper_headers = {
                "X-API-KEY": "ffd1aa778692650d8364c3ec086648238e6f6bc7",
                "Content-Type": "application/json"
            }
            serper_payload = {"q": query}
            try:
                serper_resp = requests.post(serper_url, headers=serper_headers, json=serper_payload, timeout=8)
                if serper_resp.status_code == 200:
                    serper_data = serper_resp.json()
                    organic = serper_data.get("organic", [])
                    if organic and isinstance(organic, list):
                        snippet = organic[0].get("snippet")
                        if snippet:
                            return snippet, "WebSearchAgent - Serper"
                    answer_box = serper_data.get("answerBox", {})
                    if answer_box:
                        answer = answer_box.get("answer") or answer_box.get("snippet")
                        if answer:
                            return answer, "WebSearchAgent - Serper"
                else:
                    logger.warning(f"Serper API returned status {serper_resp.status_code}")
            except Exception as se:
                logger.error(f"Serper API error: {se}")
            return "No concise web answer found.", "WebSearchAgent - None"
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Web search error: {e}", "WebSearchAgent - Error"
