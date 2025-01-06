##############################################################################
# Exercise 1:
# Please navigate and find the Langchain prompt templates used for 
# RetrievalQAWithSources chain and how it is initialized.
##############################################################################
from langchain.prompts import PromptTemplate

PROMPT = PromptTemplate(
    template="Question: {question}\nAnswer: {answer}",
    input_variables=["question", "answer"]
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)