from rag_core import get_vectorstore, list_documents, answer_question

print("Initializing RAG Pipeline Components...")

vectorstore = get_vectorstore()

docs = list_documents(vectorstore)
print("Loaded documents:", ", ".join(d["name"] for d in docs) or "none")
print("Rag system created")
print("press 0 to exit")

while True:
    query = input("You: ")
    if query == "0":
        break
    if not query.strip():
        continue

    answer, sources = answer_question(vectorstore, query)
    print(f"\nAI: {answer}")
    refs = sorted({(s["file"], s["page"]) for s in sources if s["page"]})
    if refs:
        print("(Sources: " + ", ".join(f"{f} p.{p}" for f, p in refs) + ")\n")