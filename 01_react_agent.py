from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch


load_dotenv()


if __name__ == '__main__':
    
    llm = ChatOpenAI(temperature=0, model='gpt-4.1')
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools)

    result = agent.invoke({"messages": HumanMessage(content="Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")})

    print(result)
