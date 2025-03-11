import os
import streamlit as st
import base64
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Importa ChatGroq dalla libreria langchain_groq
from langchain_groq.chat_models import ChatGroq

load_dotenv()
_ = load_dotenv

os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']

# Configura la chiave API di Groq
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Assicurati di impostare la variabile d'ambiente 'GROQ_API_KEY' con la tua chiave API di Groq.")

# -------------------------------
# Funzione di inizializzazione della chain con cache per Streamlit
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_qa_chain():
    # 1. Crea gli embeddings utilizzando un modello Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Inizializza ChromaDB in locale per il vectorstore
    persist_directory = "./chroma_db"
    docsearch = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

    # 3. Configura il modello LLM utilizzando ChatGroq
    llm = ChatGroq(temperature=0.7, groq_api_key=api_key, model_name="llama3-70b-8192")

    # 4. Definisce una prompt template per simulare le parole di JFK in prima persona
    prompt_template = PromptTemplate(
        template = """
You are John F. Kennedy, the 35th President of the United States. You must embody my persona, speaking in the first person with the charisma, wit, and eloquence for which He was known.

### **Guidelines for Your Responses:**
- **Authentic Voice & Style:** Speak with the optimism, inspiration, and determination that defined his speeches.
- **Historical Perspective:** Respond based solely on the information available up until November 22, 1963. He has no knowledge of events beyond that date. If asked about the future, you may speculate based on his beliefs, values, and policy priorities.
- **Presidential Knowledge:** your responses should demonstrate his expertise in domestic and international affairs, including:
  - The Cold War, nuclear tensions, and diplomacy with the Soviet Union.
  - The Cuban Missile Crisis and my doctrine of peaceful resolution through strength.
  - The Space Race and my commitment to landing a man on the Moon.
  - Civil rights, economic policy, and my vision for a stronger, united America.
  - My personal background, family, and political career, reflecting my experiences as a senator, World War II veteran, and president.
- **Avoid Modern References:** Do not acknowledge events, technologies, or people beyond his era (1917-1963). If asked, respond as if speaking from his own historical moment.

Follow these behavioral rules:
- If the user asks a question, answer as if you are speaking to him, only to him.
- Don't be too verbose. Answer the questions of the user in an efficient way.
- Don't refer to the user as only an American, he can be from every part of the world.

### **Using Context Effectively:**
The following **context** contains relevant information that may assist in formulating your response. When applicable, integrate this knowledge seamlessly into your answers, as if recalling it from memory. Ensure that any additional historical facts align with my known perspectives.

---
**Context:**
{context} 
---
**Chat History:**
{chat_history}
---
**Question:** {question}  
**Response:**  
""",
        input_variables=["context","chat_history",  "question"],
    )

    # 5. Crea la chain RetrievalQA combinando il retriever (ChromaDB) e il modello ChatGroq
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template},
    )

    return qa_chain

# Carica la chain (con cache, per velocizzare l'avvio)
qa_chain = load_qa_chain()
if qa_chain is None:
    st.stop()

def get_image_base64(image_path):
    """Converte un'immagine locale in base64 per usarla in HTML."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Lista di immagini per lo slideshow (con didascalie)
images = [
    ("immagini/John_F._Kennedy,_White_House_color_photo_portrait.jpg", "Kennedy's Photo Portrait"),
    ("immagini/birthplace-John-F-Kennedy-Massachusetts-Brookline.jpg", "Kennedy's Birthplace"), 
    ("immagini/John-F-Kennedy-graduation-attire.jpg", "Kennedy's graduation"),
    ("immagini/Button-campaign-John-F-Kennedy-1960.jpg", "Kennedy's Button Campaign"), 
    ("immagini/JFK_speech_Ich_bin_ein_berliner_1.jpg", "Speech in berlin"),
    ("immagini/President_Kennedy_signs_Nuclear_Test_Ban_Treaty,_07_October_1963.jpg", "Nuclear test ban treaty")
]

# Creazione dello slideshow HTML con autoplay e frecce di navigazione
slideshow_html = """<style>
.slideshow-container {
  position: relative;
  max-width: 100%;
  height: 400px;
  margin: auto;
  overflow: hidden;
}

.mySlides {
  display: none;
  text-align: center;
  position: absolute;
  width: 100%;
  height: 100%;
  transition: opacity 1s ease-in-out;
}

.mySlides img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 10px;
}

.caption {
  color: #ffffff;
  font-size: 18px;
  padding: 10px;
  text-align: center;
  background-color: rgba(0, 0, 0, 0.5);
  border-radius: 5px;
  position: absolute;
  bottom: 10px;
  left: 50%;
  transform: translateX(-50%);
  width: 80%;
}

.prev, .next {
  cursor: pointer;
  position: absolute;
  top: 50%;
  width: auto;
  padding: 10px;
  margin-top: -22px;
  color: white;
  font-weight: bold;
  font-size: 20px;
  background-color: rgba(0, 0, 0, 0.5);
  border-radius: 5px;
  user-select: none;
  transition: 0.3s;
}
.next { right: 10px; }
.prev { left: 10px; }
.prev:hover, .next:hover { background-color: rgba(0, 0, 0, 0.8); }

/* Indicatori (dots) */
.dots-container {
  text-align: center;
  position: absolute;
  bottom: 5px;
  width: 100%;
}
.dot {
  height: 12px;
  width: 12px;
  margin: 0 5px;
  background-color: rgba(255, 255, 255, 0.5);
  border-radius: 50%;
  display: inline-block;
  cursor: pointer;
  transition: background-color 0.3s;
}
.active, .dot:hover { background-color: white; }
</style>

<div class="slideshow-container">
"""
for i, (image, caption) in enumerate(images):
    image_base64 = get_image_base64(image)
    slideshow_html += f'''
    <div class="mySlides">
      <img src="data:image/jpeg;base64,{image_base64}">
      <div class="caption">{caption}</div>
    </div>
    '''

slideshow_html += """
<a class="prev" onclick="plusSlides(-1)">❮</a>
<a class="next" onclick="plusSlides(1)">❯</a>
<div class="dots-container">
"""
for i in range(len(images)):
    slideshow_html += f'<span class="dot" onclick="currentSlide({i})"></span>'
slideshow_html += """
</div>
</div>

<script>
var slideIndex = 0;
var slides = document.getElementsByClassName("mySlides");
var dots = document.getElementsByClassName("dot");

function showSlides() {
  for (var i = 0; i < slides.length; i++) {
    slides[i].style.display = "none";
    slides[i].style.opacity = "0";
  }
  for (var i = 0; i < dots.length; i++) {
    dots[i].classList.remove("active");
  }

  slides[slideIndex].style.display = "block";
  setTimeout(() => { slides[slideIndex].style.opacity = "1"; }, 100);
  dots[slideIndex].classList.add("active");
}

function plusSlides(n) {
  slideIndex = (slideIndex + n + slides.length) % slides.length;
  showSlides();
}

function currentSlide(n) {
  slideIndex = n;
  showSlides();
}

// Avvia l'autoplay ogni 7 secondi

showSlides();
</script>
"""

# Lista dei principali successi (achievements) di JFK
achievements = [
    {
        "title": "🏛️ The Moon Landing Initiative",
        "content": "In 1961, President John F. Kennedy challenged the nation to land a man on the Moon and return him safely to Earth before the decade was out. This ambitious vision led to the Apollo program and, ultimately, the 1969 Moon landing."
    },
    {
        "title": "☮️ The Cuban Missile Crisis",
        "content": "In October 1962, his administration successfully negotiated the removal of Soviet nuclear missiles from Cuba, avoiding a global nuclear catastrophe. This crisis underscored the importance of diplomacy and firm leadership."
    },
    {
        "title": "✊ Civil Rights Advocacy",
        "content": "Kennedy took a strong stance on civil rights, advocating for equal treatment of all Americans. His administration proposed the Civil Rights Act of 1964, which laid the foundation for ending segregation and racial discrimination."
    },
    {
        "title": "🌍 The Peace Corps",
        "content": "He established the Peace Corps in 1961, inspiring thousands of young Americans to volunteer and provide aid across the globe. This program continues to promote international goodwill and cooperation."
    },
    {
        "title": "📈 Economic Growth and Tax Cuts",
        "content": "His economic policies stimulated growth through tax reductions and investment in infrastructure. These measures contributed to one of the strongest periods of economic expansion in American history."
    }
]

# -------------------------------
# Interfaccia utente con Streamlit
# -------------------------------
st.set_page_config(page_title="Chatbot JFK in Prima Persona", page_icon="🇺🇸", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #0034ff, #ff0000);
        background-attachment: fixed;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Inizializza la cronologia della chat e il flag per la visibilità della chat in session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_visible" not in st.session_state:
    st.session_state.chat_visible = True
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

st.title("Chatbot RAG - John F. Kennedy Simulation")
#st.markdown("Fai una domanda sulla mia vita e risponderò come se fossi John F. Kennedy, parlando in prima persona.")

# Layout a tre colonne: Chat | Media Storici | Achievements
col_chat, col_media, col_ach = st.columns([3, 1.5, 1.5])

# --- Colonna Sinistra: Interfaccia Chat ---
with col_chat:
    st.header("Chat Interface")
    with st.form(key="query_form", clear_on_submit=True):
        user_input = st.text_input("")
        submit_button = st.form_submit_button(label="Invia")
    
    if submit_button and user_input:
        # Recupera gli ultimi 3 messaggi dalla cronologia, formattandoli in una stringa
        last_three = st.session_state.chat_history[-3:]
        # Formatta: "Mittente: messaggio"
        chat_history_str = "\n".join([f"{sender}: {msg}" for sender, msg in last_three])
        
        with st.spinner("I'm thinking..."):
            try:
                # Il metodo run riceve il "question" e il "chat_history" aggiuntivo.
                # Il parametro "context" verrà automaticamente popolato dal retriever.
                answer = qa_chain.run(question=user_input, chat_history=chat_history_str)
            except Exception as e:
                answer = f"Si è verificato un errore durante l'elaborazione: {e}"
        
    
    # Visualizza la chat solo se abilitata
    if st.session_state.chat_visible:
        # Utilizza il nuovo componente st.chat_message per ogni messaggio
        for speaker, message in st.session_state.chat_history:
            if speaker == "Utente":
                mess = st.chat_message("user")
                mess.write(message)
            else:
                mess = st.chat_message("assistant")
                mess.write(message)
    
   
            

    # Aggiungi animazioni ai messaggi della chat con JavaScript in un componente HTML personalizzato
    st.components.v1.html(
        """
        <style>
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        .chat-msg-animated {
            animation: fadeIn 1s ease-in-out;
        }
        </style>
        <script>
        // Attende che i messaggi della chat siano stati renderizzati e aggiunge la classe di animazione
        setTimeout(function(){
            const chatElements = document.querySelectorAll('[data-testid="stChatMessage"]');
            chatElements.forEach(el => {
                el.classList.add("chat-msg-animated");
            });
        }, 500);
        </script>
        """, height=0
    )
with col_media:
    with st.container():
        st.markdown("### 📜 Historical Media", unsafe_allow_html=True)
        
        # Slideshow delle immagini storiche
        st.components.v1.html(slideshow_html, height=400)

        # Aggiungi uno spazio vuoto per ridurre la distanza
       
with col_ach:
    with st.container():    
        st.markdown("### 📜 Some of the achievements of John F. Kennedy", unsafe_allow_html=True)

        # Crea una lista puntata per gli achievements
        for achievement in achievements:
            # Usa st.expander per mostrare la spiegazione dettagliata quando cliccato
            with st.expander(achievement["title"]):
                st.write(achievement["content"])
                    