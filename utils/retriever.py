def get_relevant_clauses(query, vectorstore, k=6):
    retriever = vectorstore.as_retriever(search_type="similarity", k=k)
    return retriever.get_relevant_documents(query)