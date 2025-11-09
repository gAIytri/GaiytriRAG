from rag_chain import get_rag_chain

print("🚀 Initializing Gaiytri RAG System...")
qa = get_rag_chain()
print("✅ Ready! Ask questions about Gaiytri LLC.\n")
print("Type 'quit' or 'exit' to stop.\n")

while True:
    q = input("\n💬 Ask: ")
    if q.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break
    if not q.strip():
        continue

    print("\n🤖 ", qa.invoke(q))