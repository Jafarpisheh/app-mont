from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
load_dotenv()




class ChatBot:
    def __init__(self):
        self.pdf_dir = 'data/pdfs'
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        self.llm_model = "llama3.1:8b"
        self.context_data = self.__load_pdfs()
        self.rag_chatbot = self.__create_rag()


    def get_response(self, user_message):
        return self.rag_chatbot({"question": user_message})['answer']

        

    def __load_pdfs(self):
        pdf_texts = []
        for file_name in os.listdir(self.pdf_dir):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(self.pdf_dir, file_name)
                loader = PyPDFLoader(file_path)
                pdf_texts.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(pdf_texts)


    
    def __create_rag(self):
        retriever = self._get_retriever()
        chat_model = self.__get_chat_model()

        pre_prompt = self.__get_pre_prompt()
        memory = self.__get_chat_history(pre_prompt)
        return ConversationalRetrievalChain.from_llm(
                    llm = chat_model,
                    chain_type = "stuff",
                    retriever = retriever,
                    memory = memory
                )
    
    def _get_retriever(self):
        db = FAISS.from_documents(self.context_data, embedding = self.embedding_model)
        # db = Chroma.from_documents(context_data, embedding = self.embedding_model)
        return db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    def __get_chat_model(self):
        # return ChatOpenAI(temperature = 0.4, model_name = self.llm_model)
        return ChatOllama(model = self.llm_model)

    def __get_pre_prompt(self):
        return SystemMessage(content =
                                """You are an AI assistant. Your role is to provide accurate, concise, and relevant answers.
                                - Your information is based solely on the retrieved context. 
                                - You are forbidden to mention the source of your information in the answers. 
                                - Do not mention the retrieved context, documents, or any sources explicitly.
                                - If the retrieved context does not provide the required information, simply state, "I do not have information on this," without elaborating further.
                                - Respond directly to the user's question without referencing how the answer was obtained.
                                - Do not use phrases like "Based on the context provided" or "It's mentioned that."
                                - Avoid referring to "documents," "context," or any process involved in generating the answer.
                                - Avoid fabricating or inferring information that is not certain.
                                - Always aim for clarity and precision in your responses.""")
    
    def __get_chat_history(self, pre_prompt):
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        memory.chat_memory.add_message(pre_prompt)
        return memory
        

class ChatBotOpenAI:
    def __init__(self):
        self.pdf_dir = 'data/pdfs'
        self.embedding_model = OpenAIEmbeddings()
        self.llm_model = 'gpt-4'
        context_data = self.__load_pdfs()
        self.rag_chatbot = self.__create_rag(context_data)


    def get_response(self, user_message):
        return self.rag_chatbot({"question": user_message})['answer']

        

    def __load_pdfs(self):
        pdf_texts = []
        for file_name in os.listdir(self.pdf_dir):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(self.pdf_dir, file_name)
                loader = PyPDFLoader(file_path)
                pdf_texts.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(pdf_texts)


    
    def __create_rag(self, context_data):
        db = FAISS.from_documents(context_data, embedding = self.embedding_model)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        chat_model = ChatOpenAI(temperature = 0.4, model_name = self.llm_model)
        system_message = SystemMessage(content =
                                       """You are an AI assistant. Your role is to provide accurate, concise, and relevant answers.
                                        - Your information is based solely on the retrieved context. 
                                        - You are forbidden to mention the source of your information in the answers. 
                                        - Do not mention the retrieved context, documents, or any sources explicitly.
                                        - If the retrieved context does not provide the required information, simply state, "I do not have information on this," without elaborating further.
                                        - Respond directly to the user's question without referencing how the answer was obtained.
                                        - Do not use phrases like "Based on the context provided" or "It's mentioned that."
                                        - Avoid referring to "documents," "context," or any process involved in generating the answer.
                                        - Avoid fabricating or inferring information that is not certain.
                                        - Always aim for clarity and precision in your responses.""")
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        memory.chat_memory.add_message(system_message)
        return ConversationalRetrievalChain.from_llm(
                    llm = chat_model,
                    chain_type = "stuff",
                    retriever = retriever,
                    memory = memory
                )

    def __get_prompts_templates(self):
        prompt_relevance = """
        You are classifying documents to know if this question is related to the parenting, babies, kids or toddlers or any orther topic that is in the context. Consider the chat history, if it is, answer yes, otherwise, answer no. 
        Here are some examples:
        Question: Knowing this followup history: What games can I play with my toddler?, classify this question: explain more?
        Expected Response: Yes

        Question: Knowing this followup history: What games can I play with my toddler?, classify this question: What is the capital of Germany.
        Expected Response: No

        Knowing this followup history: {chat_history}, classify this question: {question}
        """
        prompt_retrieve_query = """
        Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. 
        The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. 
        So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.
        Chat history: {chat_history}
        Question: {question}
        """
        # You are a trustful assistant of the parents. You are answering parenting related questions only based on the information provided in the context.
        prompt_retrieve = """
        You are an assistant for the parents. The source of your knowledge is only the provided context and your chat history with the user. Do not disclose the source of your information in any way.
        You are answering parenting related questions only based on the information provided in the context.
        If you do not know the answer to a question or the answer does not exist in the context, you truthfully say you do not know. 
        Read the discussion to get the context of the previous conversation. 
        In the chat discussion, you are referred to as "system". The user is referred to as "user".
        Discussion: {chat_history}
        Here's the context which is the only source of your information: {context}
        These are some important notes on how to answer the question:
            - Just give the straight answer without writing what is it based on
            - do not repeat the question
            - do not start with something like: the answer to the question
            - do not add "AI" or "Sure" in front of your answer
            - do not say: here is the answer
            - do not mention the context or the question
            - do not say what your answer is based on
            - do not answer any question about the source of your information
            - do not say based on the provided context
            - Do not mention anything about the fact that a context is provided to you!
        answer this question: {question}
        """

        return prompt_relevance, prompt_retrieve_query, prompt_retrieve
    
