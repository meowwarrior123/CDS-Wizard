from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import *
# from tools import theTools <- for tools integration

# TODOS:
# eliminate markup formatting
# specify prompt template further

load_dotenv()

class responseFormat(BaseModel):
    # response format categories go here
    advice_with_gpa: str
    advice_with_extracurriculars: str
    advice_with_major: str
    general_advice: str




# put model as param!!!
# llm1 = ChatOpenAI(model="o4-mini-2025-04-16")
llm2 = ChatAnthropic(model="claude-3-5-sonnet-20240620")
parser = PydanticOutputParser(pydantic_object=responseFormat)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a college counsellor helping students prepare for college admissions. 
            answer to the user's query carefully with thoughtful considerations.
            Provide NO OTHER TEXT. STRICTLY follow this format when wrapping your plain-text output: \n{format_instructions}
            """,

        ),
        ("placeholder", "{chat_history}"),
        ("human","{query}"), # "human", "{param1} {param2}..."
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

toolSet=[
    get_pdf_content,
    get_url_of_college_cdsData,
]

invokedAgent = create_tool_calling_agent(
    llm = llm2,
    prompt = prompt,
    tools=toolSet
)

executor = AgentExecutor(agent=invokedAgent,tools=toolSet,verbose=True)
collegeInput = input('what college would you like to look at?: ')
query = f'give me advice on applying to {collegeInput}.'
rawResponse = executor.invoke({"query":query})

try:
    cleanResponse = parser.parse(rawResponse.get("output")[0]["text"])
    # we can use cleanResponse.category for a cleaner output
    print(cleanResponse)
except Exception as e:
    print("Error occured while parsing response: ",e,"Raw response: {rawResponse}")