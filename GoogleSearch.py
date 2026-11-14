import logging
import os

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

from Config import *
from CustomAgentExecutor import SafeReActSingleInputOutputParser

os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

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

            search_results = gsearch(question, num_results=5, advanced=True)
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

    prompt = PromptTemplate.from_template(template=template)
    chain = prompt | llm | StrOutputParser()
    return call_with_rate_limit_retry(chain.invoke, {"question": question, "results": results})


def get_url_content(url):
    """
    Get content from given url.
    @param url: the given url
    """
    headers = build_http_headers()
    proxies = resolve_http_proxies(proxy)

    try:
        response = requests.get(url.strip(), headers=headers, proxies=proxies, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        texts = soup.get_text(separator="\n", strip=True)
        return texts[:320000]
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
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )
    res = call_with_rate_limit_retry(agent_executor.invoke, {"input": question})
    return res.get("output", str(res)) if isinstance(res, dict) else str(res)
