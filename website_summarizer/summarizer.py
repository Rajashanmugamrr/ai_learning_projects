import re
# import textwrap
from dataclasses import dataclass
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from openai import OpenAI


DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}


@dataclass
class ExtractedPage:
    url: str
    title: str
    text: str
    sections: List[Dict[str, Any]]
    source: str


class WebsiteContentExtractor:
    """Extract and clean website content using Playwright first, then requests fallback."""

    def __init__(self, timeout: int = 60, headers: Dict[str, str] | None = None):
        self.timeout = timeout
        self.headers = headers or DEFAULT_HEADERS

    def extract(self, url: str) -> ExtractedPage:
        html = self._fetch_with_playwright(url)
        source = "playwright"
        if html is None:
            html = self._fetch_with_requests(url)
            source = "requests"

        if html is None:
            raise RuntimeError("Could not fetch the page content from the target website.")

        soup = BeautifulSoup(html, "html.parser")
        title = self._extract_title(soup)
        cleaned = self._clean_html(soup)
        text = self._normalize_text(cleaned)
        sections = self._extract_sections(soup)

        if not text.strip():
            raise ValueError("No readable text content was found on the page.")

        return ExtractedPage(url=url, title=title, text=text, sections=sections, source=source)

    def _fetch_with_playwright(self, url: str) -> str | None:
        try:
            from playwright.sync_api import sync_playwright
        except Exception:
            return None

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(user_agent=self.headers["User-Agent"])
                page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                page.wait_for_load_state("networkidle", timeout=self.timeout * 1000)
                html = page.content()
                browser.close()
                return html
        except Exception:
            return None

    def _fetch_with_requests(self, url: str) -> str | None:
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        title = soup.title.get_text(" ", strip=True) if soup.title else "Untitled page"
        return title

    def _clean_html(self, soup: BeautifulSoup) -> str:
        for tag in soup(["script", "style", "noscript", "svg", "iframe", "button", "form"]):
            tag.decompose()

        for selector in ["nav", "header", "footer", "aside", "[role='navigation']", "[aria-label*='menu']"]:
            for tag in soup.select(selector):
                tag.decompose()

        return "\n".join(node.get_text(" ", strip=True) for node in soup.find_all(['article', 'main', 'section', 'p', 'li', 'h1', 'h2', 'h3', 'h4']) if isinstance(node, (Tag, NavigableString)))

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())

    def _extract_sections(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        sections = []
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = heading.get_text(" ", strip=True)
            if not text:
                continue

            paragraphs = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    break
                if getattr(sibling, 'get_text', None):
                    content = sibling.get_text(" ", strip=True)
                    if content:
                        paragraphs.append(content)

            if paragraphs or text:
                sections.append({"heading": text, "content": " ".join(paragraphs[:8])})

        return sections[:10]


class ChunkedSummarizer:
    """Generate summaries using the local Ollama-compatible Llama model."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")

    def summarize(self, page: ExtractedPage) -> str:
        try:
            chunked_text = self._prepare_chunks(page)
            if not chunked_text:
                raise ValueError("No text was prepared for summarization.")

            if len(chunked_text) == 1:
                return self._generate_summary(page.title, page.text, page.sections, chunked_text[0], is_combined=False)

            chunk_summaries = []
            for index, chunk in enumerate(chunked_text, start=1):
                chunk_summaries.append(self._generate_summary(page.title, chunk, page.sections, chunk, is_combined=False, chunk_index=index, total_chunks=len(chunked_text)))

            combined_prompt = self._build_combined_prompt(page, chunk_summaries)
            return self._llm_call(combined_prompt)
        except Exception as exc:
            return f"Summary generation failed: {exc}"

    def _prepare_chunks(self, page: ExtractedPage) -> List[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", page.text) if part.strip()]
        chunks = []
        current = ""

        for paragraph in paragraphs:
            if len(current) + len(paragraph) + 2 <= 1800:
                current = f"{current}\n\n{paragraph}".strip()
            else:
                if current:
                    chunks.append(current)
                current = paragraph

        if current:
            chunks.append(current)

        return chunks or [page.text[:1800]]

    def _generate_summary(self, title: str, content: str, sections: List[Dict[str, Any]], chunk: str, is_combined: bool, chunk_index: int | None = None, total_chunks: int | None = None) -> str:
        section_hint = "\n".join(f"- {item['heading']}: {item['content']}" for item in sections[:6])
        if chunk_index and total_chunks:
            instruction = (
                f"You are summarizing chunk {chunk_index} of {total_chunks} from the website '{title}'. "
                "Focus on the main idea, notable points, announcements, and important sections. "
                "Keep it concise and mention the section names when relevant."
            )
        else:
            instruction = (
                f"You are summarizing the website '{title}'. "
                "Focus on the main idea, notable points, announcements, and important sections. "
                "Keep it concise and mention section names when relevant."
            )

        prompt = (
            f"{instruction}\n\n"
            f"Detected sections:\n{section_hint or 'No section headings were detected.'}\n\n"
            f"Content to summarize:\n{chunk}\n\n"
            "Return a short summary with 3 to 6 bullet points and a one-line conclusion."
        )
        return self._llm_call(prompt)

    def _build_combined_prompt(self, page: ExtractedPage, chunk_summaries: List[str]) -> str:
        section_hint = "\n".join(f"- {item['heading']}: {item['content']}" for item in page.sections[:6])
        return (
            f"You are creating a final summary for the website '{page.title}'. "
            "Combine the chunk summaries into one coherent, useful summary. "
            "Keep the tone clear and concise, prefer the main content over navigation noise, and mention important sections when relevant.\n\n"
            f"Detected sections:\n{section_hint or 'No section headings were detected.'}\n\n"
            "Chunk summaries:\n"
            + "\n\n".join(f"Chunk {index + 1}:\n{summary}" for index, summary in enumerate(chunk_summaries))
            + "\n\n"
            "Return:\n"
            "1. A short overview\n"
            "2. A bullet list of the main takeaways\n"
            "3. A short note on the most important sections"
        )

    def _llm_call(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise website summarizer. Return concise, practical summaries."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc


class WebsiteSummarizer:
    """High-level orchestration layer for fetching and summarizing a website."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.extractor = WebsiteContentExtractor()
        self.summarizer = ChunkedSummarizer(model=model)

    def summarize_url(self, url: str) -> Dict[str, Any]:
        try:
            page = self.extractor.extract(url)
            summary = self.summarizer.summarize(page)
            return {
                "url": url,
                "title": page.title,
                "source": page.source,
                "sections": page.sections,
                "summary": summary,
            }
        except Exception as exc:
            return {
                "url": url,
                "title": "Unavailable",
                "source": "error",
                "sections": [],
                "summary": f"An error occurred while summarizing the website: {exc}",
            }


def summarize_website(url: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    return WebsiteSummarizer(model=model).summarize_url(url)
