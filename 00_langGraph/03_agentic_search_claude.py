# Lesson 3: Agentic Search Tools — local version (+ optional Claude step)
#
# Note: It only uses Tavily, DuckDuckGo and web scraping, so only TAVILY_API_KEY is required.
# A Claude step is added at the end to answer the question from search results.

import os
import re
import json

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient

try:
    from ddgs import DDGS                  # new package name
except ImportError:
    from duckduckgo_search import DDGS     # old name used in the course

_ = load_dotenv()

client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# %% 1. Tavily search with a generated answer
result = client.search("What is in Nvidia's new Blackwell GPU?", include_answer=True)
print(result["answer"])  # in a script, bare `result["answer"]` prints nothing

# %% 2. Regular search: DuckDuckGo + scraping
city = "San Francisco"

query = f"""
    what is the current weather in {city}?
    Should I travel there today?
    "weather.com"
"""

ddg = DDGS()


def search(query, max_results=6):
    try:
        results = ddg.text(query, max_results=max_results)
        return [i["href"] for i in results]
    except Exception as e:
        print(f"returning fallback results due to DDG error: {e}")
        return [
            "https://weather.com/weather/today/l/USCA0987:1:US",
            "https://weather.com/weather/hourbyhour/l/54f9d8baac32496f6b5497b4bf7a277c3e2e6cc5625de69680e6169e7e38e9a8",
        ]


for i in search(query):
    print(i)


def scrape_weather_info(url):
    """Scrape content from the given URL."""
    if not url:
        return None
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code != 200:
        print(f"Failed to retrieve the webpage (status {response.status_code}).")
        return None
    return BeautifulSoup(response.text, "html.parser")


url = search(query)[0]
soup = scrape_weather_info(url)

if soup and soup.body:
    print(f"Website: {url}\n\n")
    print(str(soup.body)[:5000])  # raw HTML is huge; trimmed from the course's 50000

    # extract readable text
    weather_data = [tag.get_text(" ", strip=True) for tag in soup.find_all(["h1", "h2", "h3", "p"])]
    weather_data = re.sub(r"\s+", " ", "\n".join(weather_data))
    print(f"Website: {url}\n\n")
    print(weather_data)

# %% 3. Agentic search: Tavily returns clean content directly
result = client.search(query, max_results=1)
data = result["results"][0]["content"]
print(data)

# The course assumes `data` is JSON-like. That's not always true today,
# so pretty-print it only if it parses.
try:
    from pygments import highlight, lexers, formatters

    parsed_json = json.loads(data.replace("'", '"'))

    with open("weather.json", "w") as f:
        json.dump(parsed_json, f, indent=4)
    print("Saved weather.json")

    formatted_json = json.dumps(parsed_json, indent=4)
    print(highlight(formatted_json, lexers.JsonLexer(), formatters.TerminalFormatter()))
except (json.JSONDecodeError, ValueError):
    print("(Result is plain text, not JSON — skipping pretty-print.)")

# %% 4. Optional: let Claude answer the question from the search results
if os.environ.get("ANTHROPIC_API_KEY"):
    from anthropic import Anthropic

    claude = Anthropic()
    search_results = client.search(f"current weather in {city}", max_results=3)
    context = "\n\n".join(r["content"] for r in search_results["results"])

    resp = claude.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=512,
        system="Answer using only the provided search results. Say if they are insufficient.",
        messages=[{
            "role": "user",
            "content": f"Search results:\n{context}\n\nQuestion: What is the current weather in "
                       f"{city}, and should I travel there today?",
        }],
    )
    print("\nClaude's answer:\n", "".join(b.text for b in resp.content if b.type == "text"))