import os
import wikipedia
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document  # Per creare documenti personalizzati

def load_pdf_documents(pdf_directory):
    """Carica tutti i documenti PDF dalla directory specificata."""
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith(".pdf")]
    documents = []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            documents.extend(docs)
            print(f"Caricato PDF: {pdf_file}")
        except Exception as e:
            print(f"Errore nel caricare il PDF {pdf_file}: {e}")
    return documents

def load_wikipedia_pages(page_titles):
    """Recupera pagine da Wikipedia date le relative voci e crea documenti LangChain."""
    wiki_documents = []
    for title in page_titles:
        try:
            # Imposta la lingua, se necessario (es. 'it' per italiano)
            wikipedia.set_lang("it")
            page = wikipedia.page(title)
            # Crea un documento con il contenuto della pagina e metadata che indicano la fonte
            doc = Document(page_content=page.content, metadata={"source": f"Wikipedia: {title}"})
            wiki_documents.append(doc)
            print(f"Caricata pagina Wikipedia: {title}")
        except Exception as e:
            print(f"Errore nel caricare la pagina Wikipedia '{title}': {e}")
    return wiki_documents

def main():
    # 1. Carica documenti PDF
    pdf_directory = "documenti"  # modifica questo percorso secondo le tue esigenze
    pdf_documents = load_pdf_documents(pdf_directory)

    # 2. Carica pagine da Wikipedia
    wikipedia_titles = [
        "John F. Kennedy"
    ]
    wiki_documents = load_wikipedia_pages(wikipedia_titles)

    # 3. Combina tutti i documenti
    documents = pdf_documents + wiki_documents
    if not documents:
        print("Nessun documento caricato. Controlla i percorsi e i titoli delle pagine.")
        return

    # 4. Suddividi i documenti in chunk per una migliore granularità
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_chunks = text_splitter.split_documents(documents)
    print(f"Documento suddiviso in {len(docs_chunks)} chunk.")

 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 6. Ingerisci i chunk nel database Chroma (persistente)
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(docs_chunks, embeddings, persist_directory=persist_directory)
    print("Database Chroma creato/aggiornato con successo.")

   
if __name__ == "__main__":
    main()
