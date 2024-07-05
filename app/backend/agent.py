from typing import List
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_openai import AzureChatOpenAI
from tools import HoasApartment, ImmigrationRP, StudyProgramme


def get_agent_tools() -> List[Tool]:
    """
    Forms the tools needed by the Finnish studying agent.
    The tools contain:
        Apartment assistant
        Resident permit assistant
        Study programme selection assistant
        DuckDuckGo search assistant.
    Returns:
        A list of langchain Tools.
    """
    apartment_apply = HoasApartment()
    RP_apply = ImmigrationRP()
    study_programme = StudyProgramme()
    ddg_search = DuckDuckGoSearchRun()

    tools = [
        Tool(
            name="Apartment assistant",
            func=apartment_apply.run,
            description="""
            Useful when you need to answer questions related to apply an apartment in Finland.
            """
        ),
        Tool(
            name="Resident permit assistant",
            func=RP_apply.run,
            description="""
            Useful when you need to answer questions related to apply a Finnish resident permit .
            """
        ),
        Tool(
            name="Study programme selection assistant",
            func=study_programme.run,
            description="""
            Useful when you need to answer questions related to select a study programme in Finland.
            """
        ),
        Tool(
            name="DuckDuckGo Search",
            func=ddg_search.run,
            description="Useful to browse information from the Internet."
        )

    ]
    return tools


def get_prompt() -> str:
    """
    Retrieves the prompt template for the agent for studying in Finland.
    Returns:
        The prompt template.
    """
    prompt = """
    You are an assistant helping international students who want to study in Finland.
    Answer the following questions as best you can, in a kind and informative manner.
    If the query is related to applying for an apartment, use the tool "Apartment assistant".
    If the query is related to a resident permit, use the tool "Resident permit assistant".
    If the query is related to study programmes, use the tool "Study programme selection assistant".
    If the question is about you, feel free to .
    If you are uncertain or there is no matching answer and the question is related to studying in Finland, use the 
    tool "DuckDuckGo Search" to search for information on the web.
    If the question is not at all related to studying in Finland, the needs of international students nor about you, 
    apologetically inform that you are meant to answer questions related to studying in Finland.
    If you encounter an invalid input or get stuck in a loop, break out by providing a helpful statement or asking a clarifying question.

    Answer the following questions as best you can. You have access to the following tools:
    [{tools}]
    
    Use the following format:
    
    Question: the input question you must answer
    
    Thought: you should always think about what to do
    
    Action: the action to take, should be one of [{tool_names}]
    
    Action Input: the input to the action
    
    Observation: the result of the action
    
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    
    Thought: I now know the final answer
    
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    
    Thought:{agent_scratchpad}
    """

    return prompt


def create_agent(llm: AzureChatOpenAI) -> AgentExecutor:
    """
    Creates an LLM agent for the given AzureChatOpenAI instance with tools for studying in Finland.
    Args:
        llm: The AzureChatOpenAI instance used in the agent.
    Returns:
        An agent executor that can be queried with questions regarding studying in Finland.
    """
    tools = get_agent_tools()
    prompt_template = ChatPromptTemplate.from_template(get_prompt())
    # prompt_template = hub.pull("hwchase17/react")
    memory = ConversationBufferWindowMemory(k=1)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)

    return agent_executor
