import asyncio
import os

from pprint import pprint

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent, BrowserConfig
from browser_use.agent.views import AgentHistoryList
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContextConfig

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
	raise ValueError('GEMINI_API_KEY is not set')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

browser = Browser(
	config=BrowserConfig(
		new_context_config=BrowserContextConfig(
			viewport_expansion=0,
		)
	)
)


async def run_search():
    agent = Agent(
		task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
		llm=llm,
		max_actions_per_step=4,
		browser=browser,
  		save_conversation_path="./logs/conversation",
	)
    history: AgentHistoryList = await agent.run(max_steps=25)
    print('Final Result:')
    pprint(history.final_result(), indent=4)
    
    print('\nErrors:')
    pprint(history.errors(), indent=4)
    
    print('\nModel Outputs:')
    pprint(history.model_actions(), indent=4)
    
    print('\nThoughts:')
    pprint(history.model_thoughts(), indent=4)
    history.save_to_file('./history1.json')
    await browser.close()


if __name__ == '__main__':
	asyncio.run(run_search())
