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
    overview_of_research_opportunities_at_target_school: str
    application_timeline: str
    advice_with_gpa: str
    advice_with_coursework: str
    advice_with_SAT_or_ACT: str
    extracurriculars_analysis_and_advice: str
    advice_with_major: str
    considerations_when_applying: str
    detailed_advice_for_writing_commonAppEssay: str
    advice_to_strengthen_admission_profile: str
    overall_fit: str
    similar_schools_with_easier_admission_difficulty: str
    similar_schools_with_same_admission_difficulty: str
    similar_schools_with_harder_admission_difficulty: str
    tools_used: list[str]


# put model as param!!!
llm1 = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
# llm2 = ChatAnthropic(model="claude-3-7-sonnet-latest")
parser = PydanticOutputParser(pydantic_object=responseFormat)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a experienced college counsellor helping students prepare for college admissions.
            Provide a nuanced and tailored answer. You must refrain from using markup-formatting and follow this format when wrapping your output:
            {format_instructions}
            Make sure to incorporate EVERY CRUCIAL detail in the tools' outputs, especially the essay_helper tool
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
            {SAT} 
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
    get_research_by_specific_college,
    get_college_cdsData,
    get_commonapp_essay_tips,
]

invokedAgent = create_tool_calling_agent(
    llm = llm1,
    prompt = prompt,
    tools = toolSet
)

# verbose on for debugging
executor = AgentExecutor(agent=invokedAgent,tools=toolSet,verbose=True)

# test value
college = 'COLLEGE'
grade = 'GRADE'
coursework = """
COURSEWORK
"""

# sample data
unweighted_GPA = 'GPA'
satScore = 'SAT SCORE'
extracurriculars = """
EXTRACURRICULARS
"""
applicantPool = 'APPLICANT POOL'
major_interest = """MAJOR INTEREST
                    """
other_details = "none"


query = f'Give me insightful advice on applying to {college} with my current stats. Be honest, dont sugarcoat anything!'

rawResponse = executor.invoke({
    "query":query, 
    "current_grade": grade,
    "grade_in_all_coursework": coursework,
    "unweighted_GPA":unweighted_GPA, 
    "SAT":satScore, 
    "extracurriculars":extracurriculars, 
    "applicant_pool":applicantPool,
    "major_interest":major_interest, 
    "other_details": other_details,
})

# jsonText = rawResponse.get("output")[0]["text"]
# jsonText = jsonText[jsonText.find("{")+1:jsonText.find("}")]
# print(jsonText, type(jsonText))

prompt2 = ChatPromptTemplate.from_messages([
    (
        'system',
        """
        You are an admissions consultant giving insightful, detailed college application advice.
        You must follow the template/format used in the input
        You must use plain text. Use no markup when providing your answer.
        """
    ),
    (
        'user',
        """
        help me revise this feedback I received with better specificity and detail. 
        Include examples where you see fit: {admissionsAdvice}
        """
    )
])

chain = prompt2 | llm1
output2 = chain.invoke({"admissionsAdvice": rawResponse}).content
print(output2, type(output2))

# print(rawResponse.get("output"), type(rawResponse))

# try:
#     cleanResponse = parser.parse(rawResponse.get("output")[0]["text"])
#     # Can use cleanResponse.category for a cleaner output
#     print(cleanResponse)
# except Exception as e:
#     print("Error occured while parsing response: ",e,"Raw response: {rawResponse}")
