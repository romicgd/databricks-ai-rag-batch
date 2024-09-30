# Databricks AI RAG Batch processing

Application of generative AI has become more diverse. Simple chatbot RAG pattern is compemented by other scenarios. 
One of the patterns is processing large amounts of data using multiple requests to GenAI model. 
For example large volume of data legal or scientific documents can be processed in batch mode to extract relevant information as pertains to each document.
This information would be stored in a database for further processing.

## Approach

In this case in Databricks onse could use user-defined functions to process data in batch mode. 
Each user-defined function is a RAG chain. 

```python
# Define the UDF to get chat completions
def call_chain(question):
    template = """
    Assistant helps the company employees with their questions on company policies, roles. 
    Always include the source metadata for each fact you use in the response. Use square brakets to reference the source, e.g. [role_library_pdf-10]. 
    Properly format the output for human readability with new lines.
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever   , "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

    result = chain.invoke(question)
    return result

# Register the UDF
call_chain_udf = udf(call_chain, StringType())

# Apply the UDF to the DataFrame
df = questions_df.repartition(3).withColumn("answer", call_chain_udf(questions_df["question"]))
display(df)
```
