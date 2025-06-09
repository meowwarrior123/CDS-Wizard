# https://python.langchain.com/docs/integrations/chat/openai/ -> helpful documentation for langchain open ai agents
# https://python.langchain.com/docs/integrations/chat/Anthropic/ -> helpful documentation for langchain open ai agents

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import *
import json
# from tools import theTools <- for tools integration


load_dotenv()

class responseFormat(BaseModel):
    # response format categories go here
    overview_of_research_and_academic_endeavors: str
    advice_with_gpa: str
    advice_with_coursework: str
    advice_with_SAT_or_ACT: str
    advice_with_extracurriculars: str
    advice_with_major: str
    considerations_when_applying: str
    advice_for_writing_commonApp: str
    advice_to_strengthen_admission_profile: str
    tools_used: list[str]


# put model as param!!!
llm1 = ChatOpenAI(model="gpt-4o-2024-11-20")
# llm2 = ChatAnthropic(model="claude-3-7-sonnet-latest")
parser = PydanticOutputParser(pydantic_object=responseFormat)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a experienced college counsellor helping students prepare for college admissions.
            Use no extra words and plain text. You must refrain from using markup-formatting and follow this format when wrapping your output:
            {format_instructions}
            """,

        ),
        ("placeholder", "{chat_history}"),
        # "human", "{param1} {param2}..."
        (
            "human",
            '''
            {query} 
            {current_grade} 
            {grade_in_all_coursework} 
            {unweighted_GPA} 
            {SAT_or_ACT} 
            {extracurriculars} 
            {applicant_pool}
            {major_interest} 
            {other_details}
            ''',
        ), 
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

toolSet=[
    get_pdf_content,
    get_url_of_college_cdsData,
    get_research_by_specific_college,
]

invokedAgent = create_tool_calling_agent(
    llm = llm1,
    prompt = prompt,
    tools=toolSet
)

executor = AgentExecutor(agent=invokedAgent,tools=toolSet,verbose=True)
# for testing purposes (TEMP VALS):
college = 'UCLA'
grade = 'Sophomore'
coursework = str(
    {
        "algebra1":'A', 
        "algebra2":'A+', 
        "precalculus":'B-', 
        "calculus BC":'A',
        "AP Lang":'B+',
        "AP Chemistry":'A-',
        "Honors Biology":'A'
    }
)
unweighted_GPA = '3.6'
SAT_or_ACT = '1540'
extracurriculars = 'SNHS officer, food drive service club founder, school gym board member, AP Chem TA'
applicantPool = 'International'
major_interest = "applied math"
other_details = "none"


query = f'Give me detailed advice on applying to {college} with my current stats.'


rawResponse = executor.invoke({
    "query":query, 
    "current_grade": grade,
    "grade_in_all_coursework": coursework,
    "unweighted_GPA":unweighted_GPA, 
    "SAT_or_ACT":SAT_or_ACT, 
    "extracurriculars":extracurriculars, 
    "applicant_pool":applicantPool,
    "major_interest":major_interest, 
    "other_details": other_details,
})

# jsonText = rawResponse.get("output")[0]["text"]
# jsonText = jsonText[jsonText.find("{")+1:jsonText.find("}")]
# print(jsonText, type(jsonText))

print(rawResponse.get("output"), type(rawResponse))

# try:
#     cleanResponse = parser.parse(rawResponse.get("output")[0]["text"])
#     # Can use cleanResponse.category for a cleaner output
#     print(cleanResponse)
# except Exception as e:
#     print("Error occured while parsing response: ",e,"Raw response: {rawResponse}")