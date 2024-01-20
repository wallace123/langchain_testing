
from decouple import config

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader


openai_key = config("OPENAI_API_KEY")
model = ChatOpenAI(openai_api_key=openai_key, 
                temperature=0.0,
                model_name="gpt-4"
        )

template = """
Here is information related to information security: <info>{text}</info>

Please do the following:
1. Summarize the information at a high school reading level (in <summary> tags). \
The summary should not exceed 150 words.
2. Highlight key points that are relevant to cloud providers or Amazon Web Services (AWS) \
(in <highlight> tags).
"""
print(template)
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()
chain = prompt | model | output_parser

loader = WebBaseLoader("https://clerk.com/changelog/2024-01-12")
text = loader.load()
print(chain.invoke({"text": text}))