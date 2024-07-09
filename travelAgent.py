import os
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent

openai = ChatOpenAI(model="gpt-3.5-turbo")

tools = load_tools(
    ['ddg-search',
    'wikipedia'],
    llm = openai
)

agent = initialize_agent(
    tools,
    openai,
    agent = 'zero-shot-react-description',
    verbose = True
)

# print(agent.agent.llm_chain.prompt.template)

query = """
    Vou viajar para Fernando de Noronha em Setembro de 2024, durante 3 dias. Me faça um roteiro de viagem com evento que irão ocorrer na data da viagem e preço das passagens de São Paulo para Fernando de Noronha.
"""

agent.run(query)