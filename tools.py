# https://python.langchain.com/docs/integrations/chat/openai/ -> helpful documentation for langchain open ai agents
# https://python.langchain.com/docs/integrations/chat/Anthropic/ -> helpful documentation for langchain open ai agents

from langchain.tools import Tool
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults, BraveSearch
import requests, anthropic, pymupdf, json

################################################################################
#FUNCTIONS
################################################################################

# initializing LLM
load_dotenv()

# for request headers
# REDACT usr agent later
userAgent = 'Your user agent here'
headers = {
    'user-agent': userAgent
}

# might be useful some place else
def scrapePdf(pdfUrl: str):
    try: 
        source = requests.get(pdfUrl, headers=headers)
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

# IMPORTANT
# still unstable as CDS can be inside another website
# if pdf url is 'none' --> get website content using requests --> scrape links and text using beautiful soup
# --> see if either the CDS pdf/xlsx is in the website or if the actual cds is there
def attemptCdsRetrieval(links):
    print(links)
    for link in links:
        # print(f'CURRENT LINK = {link}')
        try:
            rawContent = requests.get(link, headers=headers).content
            soup = BeautifulSoup(rawContent, 'lxml')
            websiteLinks = []
            for innerLink in soup.find_all('a'):
                if innerLink.get('href') not in link:
                    websiteLinks.append(innerLink)
            potentialCdsUrls = []
            for webLink in websiteLinks: 
                if (
                    ("pdf" in webLink.text.lower()) or 
                    ("xlsx" in webLink.text.lower()) or 
                    ("common data set" in webLink.text.lower()) or 
                    ("cds" in webLink.text.lower()) or 
                    ("common-data-set" in webLink.text.lower())
                ):
                    if webLink.get('href')[0] == '/':
                        heading = 'https://'
                        cutLink = link[8:]
                        content = cutLink[:cutLink.find('/')]
                        # print('CONTENT: ',content)
                        # print("LINK: ",heading+content+webLink.get('href'))
                        potentialCdsUrls.append(heading+content+webLink.get('href'))
                    else:
                        potentialCdsUrls.append(webLink.get('href'))
            if potentialCdsUrls != []:
                return scrapePdf(potentialCdsUrls[0])
            else:
                return soup.get_text()
        except Exception as e:
            print(f'an error has occured while retrieving CDS for {link}: {e}')

            

def getCDS(collegeName: str):
    
    # retrieving CDS
    braveApiKey = "your api key here"
    search = BraveSearch.from_api_key(api_key=braveApiKey, search_kwargs={"count": 5})
    response = search.run(f"{collegeName} latest CDS data PDF")
    links = []
    for entry in response.split(','):
        if "link" in entry:
            link = entry[entry.find(":")+3:-1]
            links.append(link)
    print(links)
    pdfUrls = []
    for link in links:
        if ("pdf" in link) or ("xlsx" in link):
            pdfUrls.append(link) 
    cdsData = ''
    for url in pdfUrls:
        cdsData += str(scrapePdf(url))
    if cdsData == '' or ('blocked' in cdsData.lower()) or ('forbidden' in cdsData.lower()):
        print('attempting alternate searching...')
        try:
            alternateSol = attemptCdsRetrieval(links)
            # college score card
            scorecardApiKey = "your api key here"
            baseURL = f"https://api.data.gov/ed/collegescorecard/v1/schools?api_key={scorecardApiKey}&"

            v1 = f"school.name={collegeName}"
            v2 = "latest.admissions.admission_rate.overall"
            v3 = "latest.admissions.test_requirements"
            v4 = "latest.admissions.sat_scores.midpoint.critical_reading"
            v5 = "latest.admissions.sat_scores.midpoint.math"
            v6 = "latest.admissions.sat_scores.midpoint.writing"
            v7 = "latest.admissions.sat_scores.average.by_ope_id"
            v8 = "latest.completion.completion_rate_4yr_150_white"
            v9 = "latest.completion.completion_rate_4yr_150_black"
            v10 = "latest.completion.completion_rate_4yr_150_hispanic"
            v11 = "latest.completion.completion_rate_4yr_150_asian"
            v12 = "latest.student.enrollment.all"
            v13 = "latest.student.demographics.race_ethnicity.white"
            v14 = "latest.student.demographics.race_ethnicity.black"
            v15 = "latest.student.demographics.race_ethnicity.hispanic"
            v16 = "latest.student.demographics.race_ethnicity.asian"
            vName = "school.name"

            varUrl = f'{v1}&fields='
            allVars = [vName,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16]
            for v in allVars:
                varUrl += v + ','    
            scorecardStr = ''
            requestUrl = baseURL + varUrl[:-1]
            response = requests.get(requestUrl, headers=headers)
            parsedInfo = json.loads(response.content).get('results')
            for key, val in parsedInfo[0].items():
                scorecardStr += f"{key}: {val}\n"
            return str(alternateSol)+f'college score card data: {scorecardStr}'
        except Exception as e:  
            print(f'an error has occured trying to obtain scorecard data: {e}')
    else:
        return cdsData

    # DEPRECATED
    # prompt = f'give me PDF URL of the latest CDS for {collegeName}? provide the PDF url ONLY'
    # client = anthropic.Anthropic()
    # cdsLink = ''

    # try:
    #     response = client.messages.create(
    #         model="claude-3-5-sonnet-latest",
    #         system="just provide the url and NOTHING ELSE. DO NOT VERIFY ANYTHING",
    #         max_tokens=1024,
    #         messages=[
    #             {'role':'user', 'content':prompt}
    #         ],
    #         tools=[{
    #             "type": "web_search_20250305",
    #             "name": "web_search",
    #             "max_uses":3,
    #         }]
    #     )
    #     cdsLink = response.content[-1].text
    #     return scrapePdf(cdsLink)
    # except Exception as e:
    #     print(f'an error has occured: {e}')

def getSchoolResearch(collegeName: str):
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(
        region = "us-en",
        max_results = 3,
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
        print(f'an error has occured while getting research: {e}')
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
            print(f'an error has occured trying to extract all urls in reesarch: {e}')
    
    scrapedText += f"links extracted from documents:\n {str(linksInDoc)}"
    
    # for debugging purposes
    # with open("scraped.txt", "a") as f:
    #     f.write(scrapedText)
    
    return analyseResearch(scrapedText)

def analyseResearch(scrapedText):
    llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14")
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

def scrapeEssays(collegeName):  
    wrapper = DuckDuckGoSearchAPIWrapper(
    region = "us-en",
    max_results = 4,
    source = 'text',
    )

    search = DuckDuckGoSearchResults(
        api_wrapper = wrapper,
        output_format = "list"
    )

    result = search.invoke(f"{collegeName} sample common app essays")
    urls = []
    for i in range(len(result)):
        urls.append(result[i].get("link"))
        

    scrapedText = ''
    for i in range(len(urls)):
        try:
            sourceUrl = urls[i]
            rawContent = requests.get(sourceUrl, headers=headers).content
            soup = BeautifulSoup(rawContent, 'lxml')
            scrapedText += soup.get_text()
        except Exception as e:
            print(f'error has occured while scraping essays: {e}')

    return scrapedText + f'urls: {str(urls)}'

# debugging
# with open("scraped.txt", "a") as f:
#     f.write(scrapeEssays("johns hopkins"))

def getEssayTips(collegeName: str):

    scrapedText = scrapeEssays(collegeName)

    llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14")
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
            refer to this scraped text and give me a thoughtful analysis of 
            how to write an effective main common apps essay and supplementary essay 
            for this school: {scrapedText}. 
            """
        )
    ]

    result = llm.invoke(messages)

    return result.content


################################################################################
#TOOLS
################################################################################

get_college_cdsData = Tool(
    name = 'get_college_cdsData',
    func = getCDS,
    description = 'this tool is used to extract crucial data from the latest CDS dataset for specific colleges'
)

get_research_by_specific_college = Tool(
    name = "college_research_extractor",
    func = getSchoolResearch,
    description = "this tool is used to extract detailed information regarding research done by specific colleges"
)

get_commonapp_essay_tips = Tool(
    name = "essay_helper",
    func = getEssayTips,
    description= "ths tool is used to obtain detailed tips/insight on writing common apps essays for specific colleges"
)
################################################################################
#TESTING/DEBUGGING
################################################################################

# potential tools to add:
# potential other colleges list
# college comparison (?) 
# common app essay helper

# testing scorecard API
# requestUrl = f"https://api.data.gov/ed/collegescorecard/v1/schools?api_key={scorecardApiKey}&school.name=columbia university&fields=school.name,latest.admissions.admission_rate.overall,latest.admissions.sat_scores.average.by_ope_id"
# response = requests.get(requestUrl, headers=headers)
# print(json.loads(response.content).get('results'))



# testing attempt retrieval function
# raw = requests.get('https://apb.ucla.edu/campus-statistics/common-data-set-undergraduate-profile',headers=headers).content
# soup = BeautifulSoup(raw,'html.parser')
# for link in soup.find_all('a'):
#     print(link.text)