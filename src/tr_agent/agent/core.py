from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from tr_agent.agent.prompts import SYSTEM_PROMPT
from tr_agent.agent.tools import make_tools
from tr_agent.broker.base import BaseBroker
from tr_agent.config import settings

# ReAct prompt: el agente razona (Thought) → actúa (Action) → observa (Observation)
_REACT_TEMPLATE = (
    SYSTEM_PROMPT
    + """

Tienes acceso a las siguientes herramientas:
{tools}

Para usar una herramienta sigue EXACTAMENTE este formato:

Thought: [tu razonamiento]
Action: [nombre de la herramienta, una de: {tool_names}]
Action Input: [input en JSON]
Observation: [resultado de la herramienta]

... (repite Thought/Action/Observation tantas veces como necesites)

Thought: he completado el análisis
Final Answer: [tu resumen]

Comienza:

{input}
{agent_scratchpad}"""
)


def create_agent(broker: BaseBroker) -> AgentExecutor:
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.1,  # baja temperatura para decisiones más deterministas
    )
    tools = make_tools(broker)
    prompt = PromptTemplate.from_template(_REACT_TEMPLATE)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )


def run(broker: BaseBroker, task: str) -> str:
    executor = create_agent(broker)
    result = executor.invoke({"input": task})
    return result.get("output", "")
