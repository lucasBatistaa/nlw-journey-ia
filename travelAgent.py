import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import  bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence


llm = ChatOpenAI(model="gpt-3.5-turbo")

query = """
    Vou viajar para Fernando de Noronha em Setembro de 2024, durante 3 dias. Me faça um roteiro de viagem com evento que irão ocorrer na data da viagem e preço das passagens de São Paulo para Fernando de Noronha.
"""

def researchAgent(query, llm):
    tools = load_tools(
        [
            'ddg-search',
            'wikipedia'
        ],
        llm = llm
    )

    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt, verbose=True)
    webContext = agent_executor.invoke({'input': query})

    return webContext['output']

def loadData():
    loader = WebBaseLoader(
        web_paths = ("https://www.dicasdeviagem.com/brasil/",),
        bs_kwargs = dict(parse_only=bs4.SoupStrainer(class_=("postcontentwrap", "pagetitleloading background-imaged loading-dark"))),
    )

    docs= loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstores = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstores.as_retriever()
    
    return retriever

def getRelevantDocs(query):
    retrivier = loadData()
    relevant_documents = retrivier.invoke(query)
    
    return relevant_documents

def supervisorAgent(query, llm, webContext, relevant_documents):
    prompt_template = """
        Você é um gerente de agência de viagens. Sua resposta final deverá ser um roteiro de viagem completo e detalhado. 
        Utilize o contexto de preços de passagens, o input do usuário e também os documentos relevantes para elaborar o roteiro.
        Contexto: {webContext}
        Documentos relevantes: {relevant_documents}
        Usuário: {query}
        Assistante: 
    """

    prompt = PromptTemplate(
        input_variables= ['webContext', 'relevant_documents', 'query'],
        template = prompt_template
    )

    sequence = RunnableSequence(prompt | llm)

    response = sequence.invoke({
        'webContext': webContext, 
        'relevant_documents': relevant_documents, 
        'query': query
    })

    return response

def getResponse(query, llm):
    webContext = researchAgent(query, llm)
    relevant_documents = getRelevantDocs(query)
    response = supervisorAgent(query, llm, webContext, relevant_documents)

    return response

print(getResponse(query, llm).content)