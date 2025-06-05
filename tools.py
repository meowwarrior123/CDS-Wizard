from langchain.tools import Tool
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.tools import GoogleSerperRun
import pymupdf
import requests, anthropic

################################################################################
#FUNCTIONS
################################################################################

# initializing LLM
load_dotenv()

def scrapePdf(url: str):
    try: 
        source = requests.get(url)
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

################################################################################
#TOOLS
################################################################################

get_pdf_content = Tool(
    name = 'get_website_content',
    func = scrapePdf,
    description ='this tool is used to extract the content of a certain pdf'

)

get_url_of_college_cdsData = Tool(
    name = 'get_url_of_college_cdsData',
    func = getCDSUrl,
    description = 'this tool is used to extract the url of the latest CDS dataset for specific colleges'
)

# search = DuckDuckGoSearchRun()
# result = search.invoke("what is UVA's primary research focus?")
# print(result)