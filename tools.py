# https://python.langchain.com/docs/integrations/chat/openai/ -> helpful documentation for langchain open ai agents
# https://python.langchain.com/docs/integrations/chat/Anthropic/ -> helpful documentation for langchain open ai agents

from langchain.tools import Tool
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
import pymupdf
import requests, anthropic

################################################################################
#FUNCTIONS
################################################################################

# initializing LLM
load_dotenv()

# for request headers
# REDACT usr agent later
userAgent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'
headers = {
    'user-agent': userAgent
}

def scrapePdf(url: str):
    try: 
        source = requests.get(url, headers=headers)
        data = source.content
        doc = pymupdf.open(stream=data)
        pages = []

        for page in doc:
            text = ''
            tables = page.find_tables()
            tableContent = ''
            for table in tables:
                tableContent = str(table.extract())
            text += 'START OF TABLE WITHIN THIS PAGE: \n\n'+ tableContent + '\n\n END OF TABLE WITHIN THIS PAGE \n\n'
            text += page.get_text()
            pages.append(text)
        return pages
    except Exception as e:
        print(f"an error occured: this is most likely because of an invalid link: {e}")

def getCDSUrl(collegeName: str):
    prompt = f'what is the PDF URL of the latest CDS for {collegeName}? provide the PDF url ONLY'
    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            system="Just provide one thing only: the url",
            max_tokens=1024,
            messages=[
                {'role':'user', 'content':prompt}
            ],
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses":1,
            }]
        )
        return response.content[-1].text
    except Exception as e:
        print(f'an error has occured: {e}')

def getSchoolResearch(collegeName: str):
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(
        region = "us-en",
        max_results = 5,
        source = 'text',
        )

        search = DuckDuckGoSearchResults(
            api_wrapper = wrapper,
            output_format = "list"
        )

        result = search.invoke(f"{collegeName} faculty research")
        urls = []
        for i in range(len(result)):
            urls.append(result[i].get("link"))
    except Exception as e:
        print(f'an error has occured: {e}')
        return
    
    scrapedText = ""
    linksInDoc = []
    for url in urls:
        try:
            response = requests.get(url, headers=headers)
            htmlContent = response.text
            soup = BeautifulSoup(htmlContent, 'lxml')
            extractedText = soup.get_text()
            for link in soup.find_all('a'):
                linksInDoc.append(link.get('href'))
            scrapedText += extractedText + '\n\n'
        except Exception as e:
            print(f'an error has occured: {e}')
    
    scrapedText += f"links extracted from documents:\n {str(linksInDoc)}"
    
    # for debugging purposes
    # with open("scraped.txt", "a") as f:
    #     f.write(scrapedText)
    
    return analyseResearch(scrapedText)

def analyseResearch(scrapedText):
    llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14")
    messages = [
        (
            "system",
            """
            you are a college consultant researching faculty research in a specific school.
            You are a 1-response API. 
            Your output must strictly be provided in plain text, no markup format.
            """
        ),
        (
            "human", 
            f"""
            refer to this scraped text and give me a thoughtful analysis of this school's 
            faculty research, including research strengths and potential downsides, 
            primary research focus, and research opportunities for undergrads
            for instance: {scrapedText}. 
            """
        )
    ]
    response = llm.invoke(messages)
    return response.content


################################################################################
#TOOLS
################################################################################

get_pdf_content = Tool(
    name = 'get_website_content',
    func = scrapePdf,
    description ='MUST USE: this tool is used to parse the content of CDS dataset'

)

get_url_of_college_cdsData = Tool(
    name = 'get_url_of_college_cdsData',
    func = getCDSUrl,
    description = 'this tool is used to extract the url of the latest CDS dataset for specific colleges'
)

get_research_by_specific_college = Tool(
    name = "college_research_extractor",
    func = getSchoolResearch,
    description = "this tool is used to extract information regarding research done by specific colleges"
)