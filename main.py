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
college = 'Carnegie Mellon university'
grade = 'junior'
coursework = """
2022-2023 | Out of DistrictGrade 9 | Fall
index: Schl, Year, Tm, Grd, Crs ID, Std Course Title, Mark
975, 2022-2023, 1, 9, 2410, Korean 1A, C 
975, 2022-2023, 1, 9, 3047, Math IA, C 
975, 2022-2023, 1, 9, 40039, PE Course 1A, A 
975, 2022-2023, 1, 9, 6007, Social Science, C 
975, 2022-2023, 1, 9, 7527, String Orch A, A
975, 2022-2023, 1, 9, 8421, Tech Training F, B 

2022-2023 | Out of DistrictGrade 9 | Spring
index: Schl, Year, Tm, Grd, Crs ID, Std Course Title, Mark
975, 2022-2023, 2, 9, 2412, Korean 1B, C
975, 2022-2023, 2, 9, 3048, Math IB, C 
975, 2022-2023, 2, 9, 40040, PE Course 1B, A 
975, 2022-2023, 2, 9, 6007, Social Science, C 
975, 2022-2023, 2, 9, 7528, String Orch B, A
975, 2022-2023, 2, 9, 8421, Tech Training F, B

2023-2024 | Out of DistrictGrade 10 | Fall
index: Schl, Year, Tm, Grd, Crs ID, Std Course Title, Mark
975, 2023-2024, 1, 10, 1073, English 1A, A 
975, 2023-2024, 1, 10, 20000, Korean 2A, B
975, 2023-2024, 1, 10, 3049, Math IIA, B 
975, 2023-2024, 1, 10, 40041, PE Course 2A, A
975, 2023-2024, 1, 10, 55000, Physics In Un A, B
975, 2023-2024, 1, 10, 6303, US History A, A 
975, 2023-2024, 1, 10, 7014, Art Studio A, A

2023-2024 | Out of DistrictGrade 10 | Spring
index: Schl, Year, Tm, Grd, Crs ID, Std Course Title, Mark
975, 2023-2024, 2, 10, 1075, English 1B, A
975, 2023-2024, 2, 10, 20001, Korean 2B, C
975, 2023-2024, 2, 10, 3050, Math IIB, B 
975, 2023-2024, 2, 10, 40042, PE Course 2B, B
975, 2023-2024, 2, 10, 55001, Physics In Un B, A
975, 2023-2024, 2, 10, 6305, US History B, A
975, 2023-2024, 2, 10, 7015, Art Studio B, A

2024-2025 | University High SchoolGrade 11 | Fall
index: Schl, Year, Tm, Grd, Crs ID, Std Course Title, Mark
608, 2024-2025, 1, 11, 1152, AP Eng Lang A, B- 
608, 2024-2025, 1, 11, 3720, AP Calc AB-A, A- 
608, 2024-2025, 1, 11, 3817, AP CmpSciA Fall, A
608, 2024-2025, 1, 11, 38642, HPrincofEn A, A- 
608, 2024-2025, 1, 11, 55302, AP Physics 1/2A, C 
608, 2024-2025, 1, 11, 6253, M World Hist A, A 
608, 2024-2025, 1, 11, 7535, Jazz Ensmble 1A, A- 

2024-2025 | University High SchoolGrade 11 | Spring
index: Schl, Year, Tm, Grd, Std Course Title, Mark
608, 2024-2025, 2, 11, AP Comp sci A, B
608, 2024-2025, 2, 11, AP Calc, B
608, 2024-2025, 2, 11, AP Lang, A
608, 2024-2025, 2, 11, Jazz, A
608, 2024-2025, 2, 11, Principle of Engineering, A
608, 2024-2025, 2, 11, Modern History, A
"""

# sample data
unweighted_GPA = '3.4737'
satScore = '1480'
extracurriculars = """
Extracurricular Activities:
- Arduino Club President - Led workshops on mechanical assembly and coding.
- Private Academic Mentor - Supported students academically in core subjects.
- Volunteer - Assisted individuals with physical disabilities.
- Basketball & Volleyball Teams - Participated in district-level tournaments.

Projects:
Smart Car for the Visually Impaired
- Designed and built an Arduino-based assistive technology smart car.
- Integrated ultrasonic sensors and buzzers to detect obstacles and provide real-time audio feedback.
- Programmed in C++ and used serial monitoring for debugging.

Awards:
- Academic Excellence Award in Middle School
- Best Drummer Award - Citywide Music Festival (2022)

Skills:
- Programming: Python, Java, C++, Arduino IDE
- Tools: Visual Studio Code
- Soft Skills: Team Collaboration, Leadership

Professional Experience:
Lead Drummer - Current
- Coordinated rhythm sections during performances and rehearsals.
- Mentored junior members and contributed to musical arrangements.
- Founded a jazz band in middle school, serving as musical leader.
President, Arduino Club - November 2023 - May 2024
- Founded and led the school's Arduino club.
- Organized weekly workshops involving mechanical assembly and coding.
Private Academic Mentor - Current
- Mentoring middle school students in English and other core subjects.
- Assisted with understanding lessons, homework, and test preparation.
- Developed personalized study plans and provided academic support.
- Helped mentees build confidence and improve school performance.
Volunteer - Current
- Supported individuals with physical disabilities on a weekly basis.
- Demonstrated perseverance, patience, and compassion.
- Committed to making a positive impact in the community.
Teacher's Assistant, Bethel Church, Irvine, CA - July 2024 - Current
- Oversaw students in recess environments to ensure safety.
- Supported classroom upkeep, organized books and materials.
"""
applicantPool = 'Domestic, California Resident'
major_interest = """robotics, 
                    electrical engineering, 
                    mechanical engineering,
                    computer science,
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
