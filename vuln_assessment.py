from decouple import config

# model imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# loader and splitter imports
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# vector store imports
from langchain_community.vectorstores import Chroma

# prompt, output, history and runnables imports
# These will be used to pass data and questions to the llm
from langchain.chains import ConversationalRetrievalChain

# Set up our model
# Temperature is a value you can play with. 0.0 is the most conservative, 1.0 is the most creative
openai_key = config("OPENAI_API_KEY")
model = ChatOpenAI(openai_api_key=openai_key, 
                temperature=0.0,
                model_name="gpt-4"
        )

# Get our own data 
web_loader = WebBaseLoader(["https://www.first.org/cvss/v3.1/specification-document", 
            "https://www.first.org/cvss/v4.0/specification-document",
            "https://httpd.apache.org/security/vulnerabilities_24.html",
            "https://httpd.apache.org/docs/2.4/misc/security_tips.html"])
# I copied text from NVD and placed in local files. I could have just added the urls
# to web_loader, but I wanted to try out multiple loaders.
txt_loader = DirectoryLoader("docs/", glob="**/*.txt", loader_cls=TextLoader)
loaders = [web_loader, txt_loader]

# Load the data into a list
docs = []
for loader in loaders:
    docs.extend(loader.load())

# See how many docs were loaded and the size of each. This helps with the next set of code, splitting text.
print(f'You have {len(docs)} document(s) in your data')
for doc in docs:
    print(f'There are {len(doc.page_content)} characters in {doc.metadata["source"]}')

# Need to split the text from the documents or we'll run into token limits
# The numbers used in chunk_size and chunk_overlap were chosen by the author of this script
# Feel free to play around with these numbers for your own data
# Tool that may help during testing (EXTERNAL SITE, DON'T USE FOR SENSITIVE DATA) https://chunkviz.up.railway.app/ 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500) # chunk overlap seems to work better
docs = text_splitter.split_documents(docs)
print(len(docs))

# What are embeddings? They help the llm find relevant data easier/faster.
# https://www.cloudflare.com/learning/ai/what-are-embeddings
# Good video https://www.youtube.com/watch?v=ySus5ZS0b94
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
# Now we're going to store embeddings in a vectorstore
# There are many vectorstore databases. I chose Chroma because it was in many tutorials and seemed easiest to work with.
# For this proof of concept, just going to load in memory and not persist. If persisted, could load from_existing().
# https://python.langchain.com/docs/integrations/vectorstores
# https://python.langchain.com/docs/integrations/vectorstores/chroma
vectorstore = Chroma.from_documents(docs, embeddings)

# When we ask a question to the llm, we want to retrieve the relevant docs from our vectorstore.
# Example, if we ask about dogs, we may want docs related to canine or puppy, not skyscrappers. 
# https://python.langchain.com/docs/modules/data_connection/retrievers/vectorstore
retriever = vectorstore.as_retriever(search_kwars={"k":5}) # k is n-1 number of docs to return
# Optional: Test the retriever
#docs = retriever.get_relevant_documents("What are the metric values for Attack Complexity in CVSS v3.1?")
#print(len(docs))
#for doc in docs:
    #print(doc.metadata["source"])
    #print(doc.page_content)

chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=retriever,
            verbose=True,
)

chat_history = []
question = "Explain Attack Vectors in CVSS v3.1?"
res = chain.invoke({"question":question, "chat_history": chat_history})
print(res['answer'])
print('\n\n')
chat_history.append((question, res['answer']))
question = "What are the list of possible values for it?"
res = chain.invoke({"question":question, "chat_history": chat_history})
print(res['answer'])