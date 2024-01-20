from decouple import config

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


openai_key = config("OPENAI_API_KEY")
model = ChatOpenAI(openai_api_key=openai_key, 
                temperature=0.0,
                model_name="gpt-4"
        )
prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
output_parser = StrOutputParser()
chain = prompt | model | output_parser

print(chain.invoke({"foo": "ice cream"}))