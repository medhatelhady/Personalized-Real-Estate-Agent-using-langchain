
#import libraries
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory

import os
os.environ["OPENAI_API_KEY"] =""


personal_questions = [   
    "How big do you want your house to be?" 
    "What are 3 most important things for you in choosing this property?", 
    "Which amenities would you like?", 
    "Which transportation options are important to you?",
    "How urban do you want your neighborhood to be?",   
]


# load gpt model
#model_name = 'gpt-3.5-turbo'
#llm = ChatOpenAI(model_name=model_name, temperature=0.3, max_tokens=100)


# read and load the csv file that store homes data
loader = CSVLoader(file_path='home.csv')
docs = loader.load()


# create vector store index and query the data
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(documents=split_docs, embedding=embeddings)



# create a chat with the customer and summarize it
history = ChatMessageHistory()
history.add_user_message(f"""You are AI sales assisstant that will recommend user a home based on their answers to personal questions. Ask user {len(personal_questions)} questions""")
for i in range(len(personal_questions)):
    history.add_ai_message(personal_questions[i])
    history.add_user_message(input(personal_questions[i]+'\nanswer: '))
    
history.add_ai_message("""Now tell me a summary of a home you're considering in points""")
memory = ConversationSummaryMemory(
    llm=llm,
    chat_memory=history,
    memory_key="chat_history", 
    input_key="question",
    buffer=f"The human answered {len(personal_questions)} personal questions. Use them to extract home attributes like location, price, home area and number of rooms",
    return_messages=True)

#print(memory.summary_message_cls.content)

# create a prompt
prompt=PromptTemplate(
    template="You are an sales assistant who buy homes. Use the following pieces of retrieved context and customer prefrences to provide the customer with information about available home. Use five sentences maximum and keep the answer attractive. \nContext: {context} \nCustomer's prefernced: {chat_history} \nQuestion: {question}\nAnswer:",
    input_variables=['context', 'chat_history', 'question']
    )

# create question and answer model to retrieve answers from retrived information
chain_type_kwargs = {'prompt': prompt}

chain = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                chain_type="stuff",
                                retriever=db.as_retriever(),
                                combine_docs_chain_kwargs=chain_type_kwargs,
                                memory=memory
                            )


# take input from user
query = "as  a sales assisstant, represent the answer in attractive way"

# run the query
result = chain({"question": query})
print(result['answer'])
