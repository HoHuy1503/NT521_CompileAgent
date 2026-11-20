import itertools
import logging
import os

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.agents import Tool, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

from compile_agent.config import *
from compile_agent.agent.executor import CompilationAgentExecutor, SafeReActSingleInputOutputParser

proxy = PROXY if USE_PROXY and PROXY else None


def google_search(question):
    """
    Search relevant information with the question.
    @param question: the question that you wanted to search
    """
    results = ""
    try:
        with DDGS(proxy=proxy, timeout=20) as ddgs:
            search_results = list(ddgs.text(question, max_results=5))
        for idx, result in enumerate(search_results, start=1):
            title = result.get("title", "")
            body = result.get("body", "")
            href = result.get("href", "")
            results += f"{idx}. Title: {title}\n{idx}. Description: {body}\n{idx}. URL: {href}\n"
    except Exception as ddg_err:
        logging.warning(f"[!] DuckDuckGo search failed ({ddg_err}), trying fallback...")
        try:
            from googlesearch import search as gsearch

            try:
                search_iter = gsearch(
                    question,
                    num_results=5,
                    advanced=True,
                    timeout=10,
                    proxy=proxy,
                )
            except TypeError:
                try:
                    search_iter = gsearch(
                        question,
                        num_results=5,
                        advanced=True,
                        timeout=10,
                    )
                except TypeError:
                    search_iter = gsearch(question, num_results=5, advanced=True)
            search_results = list(itertools.islice(search_iter, 5))
            for idx, result in enumerate(search_results, start=1):
                results += (
                    f"{idx}. Title: {getattr(result, 'title', '')}\n"
                    f"{idx}. Description: {getattr(result, 'description', '')}\n"
                    f"{idx}. URL: {getattr(result, 'url', '')}\n"
                )
        except Exception as g_err:
            logging.error(f"[!] Fallback web search failed: {g_err}")
            return f"Search failed: {g_err}"

    if not results.strip():
        return "No search results found."

    template = """You are a Search Result Analysis Expert, specializing in evaluating and ranking web pages based on relevance and quality. Your task is to analyze a set of search results and identify the top 3 most relevant URLs for the given query.
    Query: {question}
    Results:
    {results}

    Output only the top 3 most relevant URLs in ranked order, without additional explanation:
    1. [Most relevant URL]
    2. [Second relevant URL]
    3. [Third relevant URL]
    """

    llm = create_chat_llm()

    prompt = PromptTemplate.from_template(template=apply_thinking_directive(template))
    chain = prompt | llm | StrOutputParser()
    try:
        raw_answer = call_with_llm_retry(
            chain.invoke, {"question": question, "results": results}
        )
        return strip_thinking_tags(raw_answer)
    except Exception as e:
        if is_fatal_llm_error(e) or is_transient_llm_error(e):
            raise
        logging.warning(f"[!] google_search ranking failed: {e}")
        return results[:4000]


def get_url_content(url):
    """
    Get content from given url.
    @param url: the given url
    """
    headers = build_http_headers()
    proxies = resolve_http_proxies(proxy)
    url_str = str(url or "").strip()
    max_download_bytes = 512 * 1024

    try:
        with requests.get(
            url_str,
            headers=headers,
            proxies=proxies,
            timeout=15,
            stream=True,
        ) as response:
            response.raise_for_status()
            content_type = (response.headers.get("Content-Type") or "").lower()
            allowed_types = (
                "text/",
                "application/xhtml",
                "application/xml",
                "application/json",
            )
            if content_type and not any(tok in content_type for tok in allowed_types):
                return f"Skipped non-text URL content ({content_type}): {url_str}"

            chunks = []
            total_bytes = 0
            for chunk in response.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                chunks.append(chunk)
                total_bytes += len(chunk)
                if total_bytes >= max_download_bytes:
                    break
            raw_bytes = b"".join(chunks)[:max_download_bytes]
            encoding = response.encoding or response.apparent_encoding or "utf-8"
            html_text = raw_bytes.decode(encoding, errors="replace")

        soup = BeautifulSoup(html_text, "html.parser")
        texts = soup.get_text(separator="\n", strip=True)
        if len(texts) > 16000:
            return texts[:16000] + "\n... (truncated)"
        return texts
    except Exception as e:
        logging.error(e)
        return f"Request url error: {url}"


def search_agent(question):
    """
    Search the web for compilation troubleshooting solutions, errors, or package names that you cannot solve offline.
    @param question: The search question.
    """
    llm = create_chat_llm()

    template = """You are a Content Retrieval and Summarization Expert. Your task is to search for relevant information, retrieve content from the top 3 URLs, and provide a comprehensive summary.

    Available tools:
    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [ {tool_names} ]
    Action Input: the input to the action, do not explain the input further
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)

    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""

    tools = [
        Tool(
            name="GOOGLE_SEARCH",
            func=google_search,
            description=google_search.__doc__,
        ),
        Tool(
            name="GET_CONTENT_FROM_URL",
            func=get_url_content,
            description=get_url_content.__doc__,
        ),
    ]

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        output_parser=SafeReActSingleInputOutputParser(),
        stop_sequence=resolve_react_stop_sequences(),
    )
    agent_executor = CompilationAgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True,
        max_iterations=6,
        max_execution_time=90,
        handle_parsing_errors=True,
    )
    try:
        res = agent_executor.invoke({"input": apply_thinking_directive(question)})
        raw_out = res.get("output", str(res)) if isinstance(res, dict) else str(res)
        return strip_thinking_tags(raw_out)
    except Exception as e:
        if is_fatal_llm_error(e) or is_transient_llm_error(e):
            raise
        logging.warning(f"[!] search_agent failed: {e}")
        return f"Search failed: {e}"
