import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import faiss
import os
from sentence_transformers import SentenceTransformer
import hashlib
import pickle
from dotenv import load_dotenv
import openai
import base64
import requests

# --- Carregar Variáveis de Ambiente ---
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    st.error("A chave da API do OpenAI não foi encontrada. Verifique o arquivo .env.")
    st.stop()

# --- Configurações Iniciais ---
FAISS_INDEX_IMAGE = "faiss_index_image.index"
FAISS_INDEX_TEXT = "faiss_index_text.index"
IMAGE_DIR = "images"
PRONTUARIOS_DATA_FILE = "prontuarios_data.pkl"


os.makedirs(IMAGE_DIR, exist_ok=True)


if os.path.exists(PRONTUARIOS_DATA_FILE):
    with open(PRONTUARIOS_DATA_FILE, 'rb') as f:
        prontuarios_data = pickle.load(f)
    
    
    for key, value in prontuarios_data.items():
        if isinstance(value, str):
            prontuarios_data[key] = {
                "prontuario_text": value,
                "imagens": []
            }
else:
    prontuarios_data = {}


@st.cache_resource
def load_text_embedding_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')  
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo de texto: {e}")
        st.stop()

@st.cache_resource
def load_image_embedding_model():
    try:
        model = SentenceTransformer('clip-ViT-B-32')  
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo de imagem: {e}")
        st.stop()

text_embedding_model = load_text_embedding_model()
image_embedding_model = load_image_embedding_model()


try:
    DIM_TEXT = text_embedding_model.get_sentence_embedding_dimension()
    st.write(f"DIM_TEXT: {DIM_TEXT}")
except Exception as e:
    st.error(f"Erro ao obter a dimensão do embedding de texto: {e}")
    st.stop()


try:
    dummy_image = Image.new('RGB', (224, 224))  
    embedding = image_embedding_model.encode([dummy_image], convert_to_numpy=True)
    if embedding is not None and hasattr(embedding, 'shape'):
        DIM_IMAGE = embedding.shape[1]
        st.write(f"DIM_IMAGE: {DIM_IMAGE}")
    else:
        st.error("Falha ao obter a dimensão do embedding de imagem. O embedding retornou None ou não possui atributo 'shape'.")
        st.stop()
except Exception as e:
    st.error(f"Erro ao obter a dimensão do embedding de imagem: {e}")
    st.stop()


def load_faiss_index(index_path, dimension):
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            if index.ntotal > 0:
                if index.d != dimension:
                    st.error(f"Dimensão do índice ({index.d}) não corresponde à dimensão esperada ({dimension})")
                    st.stop()
        except Exception as e:
            st.error(f"Erro ao ler o índice FAISS {index_path}: {e}")
            st.stop()
    else:
        try:
            index = faiss.IndexFlatL2(dimension)
            faiss.write_index(index, index_path)
        except Exception as e:
            st.error(f"Erro ao criar o índice FAISS {index_path}: {e}")
            st.stop()
    return index

index_text = load_faiss_index(FAISS_INDEX_TEXT, DIM_TEXT)
index_image = load_faiss_index(FAISS_INDEX_IMAGE, DIM_IMAGE)


def save_prontuarios_data():
    try:
        with open(PRONTUARIOS_DATA_FILE, 'wb') as f:
            pickle.dump(prontuarios_data, f)
    except Exception as e:
        st.error(f"Erro ao salvar prontuarios_data: {e}")


def add_prontuario_to_faiss(nome, idade, email, telefone, motivo_consulta, outras_doencas, medicamentos):
    prontuario_text = (
        f"Nome: {nome}\n"
        f"Idade: {idade}\n"
        f"Email: {email}\n"
        f"Telefone: {telefone}\n"
        f"Motivo da consulta: {motivo_consulta}\n"
        f"Outras doenças: {outras_doencas}\n"
        f"Medicamentos: {medicamentos}"
    )
    try:
        embedding = text_embedding_model.encode([prontuario_text], convert_to_numpy=True)
        embedding = np.array(embedding).astype('float32')
        faiss.normalize_L2(embedding)
        index_text.add(embedding)
        faiss.write_index(index_text, FAISS_INDEX_TEXT)
        
        
        prontuario_id = index_text.ntotal - 1
        prontuarios_data[prontuario_id] = {
            "nome": nome,
            "idade": idade,
            "email": email,
            "telefone": telefone,
            "motivo_consulta": motivo_consulta,
            "outras_doencas": outras_doencas,
            "medicamentos": medicamentos,
            "prontuario_text": prontuario_text,
            "imagens": []
        }
        save_prontuarios_data()
        return prontuario_id
    except Exception as e:
        st.error(f"Erro ao adicionar prontuário ao FAISS: {e}")
        st.stop()


def add_image_to_faiss(image_pil, image_name, prontuario_id):
    try:
        
        embedding = image_embedding_model.encode([image_pil], convert_to_numpy=True)
        embedding = np.array(embedding).astype('float32')
        
        
        st.write(f"Embedding shape: {embedding.shape}")
        st.write(f"DIM_IMAGE: {DIM_IMAGE}")
        
        if embedding.shape[1] != DIM_IMAGE:
            raise ValueError(f"Dimensão do embedding ({embedding.shape[1]}) não corresponde à dimensão do índice de imagem ({DIM_IMAGE})")
        
        
        faiss.normalize_L2(embedding)
        
        
        st.write("Adicionando embedding ao índice FAISS...")
        index_image.add(embedding)
        faiss.write_index(index_image, FAISS_INDEX_IMAGE)
        
        
        image_hash = hashlib.md5(image_name.encode()).hexdigest()
        image_path = os.path.join(IMAGE_DIR, f"{image_hash}.png")
        image_pil.save(image_path)
        st.write(f"Imagem salva em: {image_path}")
        
        
        prontuarios_data[prontuario_id]['imagens'].append(image_path)
        save_prontuarios_data()
        
        return index_image.ntotal - 1  
    except Exception as e:
        st.error(f"Erro ao adicionar imagem ao FAISS: {e}")
        st.stop()


def search_faiss(query_embedding, index, k=5):
    try:
        D, I = index.search(query_embedding, k)
        return I[0]
    except Exception as e:
        st.error(f"Erro ao realizar busca no FAISS: {e}")
        st.stop()


def encode_image_base64_url(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_image}"


def get_multimodal_response(prompt, image_path=None):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        
        content = [{"type": "text", "text": prompt}]
        
        
        if image_path:
            base64_image_url = encode_image_base64_url(image_path)
            content.append({"type": "image_url", "image_url": {"url": base64_image_url}})

        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": content}
            ],
            "max_tokens": 300
        }

        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        
        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content'].strip()
            return answer
        else:
            st.error(f"Erro na API: {response.status_code}, {response.text}")
            st.stop()

    except Exception as e:
        st.error(f"Erro ao obter resposta do GPT-4 Multimodal: {e}")
        st.stop()


def get_text_response(prompt, context):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """
                        Você é um assistente de inteligência artificial especializado em auxiliar dentistas a consultar informações detalhadas sobre seus pacientes. Responda de maneira clara e técnica."""},
                {"role": "user", "content": f"{context}\n\nPergunta: {prompt}"}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        answer = response.choices[0].message['content'].strip()
        return answer
    except Exception as e:
        st.error(f"Erro ao obter resposta do ChatGPT: {e}")
        st.stop()

# --- Início da Aplicação Streamlit ---
st.title("Demonstração de Funcionalidade")


tabs = st.tabs(["Prontuário e Detecção de Problemas", "Sistema RAG"])

# --- Aba 1: Prontuário e Detecção de Cáries ---
with tabs[0]:
    
    
    with st.form("prontuario_e_deteccao_form"):
        st.subheader("Prontuário Odontológico")
        nome = st.text_input("Nome")
        idade = st.number_input("Idade", min_value=0, max_value=120, step=1)
        email = st.text_input("E-mail")
        telefone = st.text_input("Telefone")
        motivo_consulta = st.text_area("Motivo da Consulta")
        outras_doencas = st.text_area("Outras Doenças")
        medicamentos = st.text_area("Medicamentos")
        
        st.markdown("---")
        
        st.subheader("Detecção de Cáries")
        uploaded_file = st.file_uploader("Faça o upload de uma foto da boca", type=["jpg", "jpeg", "png"])
        image_placeholder = st.empty()
        annotated_image_placeholder = st.empty()
        
        annotated_image = None
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_placeholder.image(image, caption='Foto Original', use_column_width=True)
            
            img_np = np.array(image)
            model = YOLO("best.pt")
            
            with st.spinner('Detectando cáries...'):
                try:
                    results = model(img_np)
                except Exception as e:
                    st.error(f"Erro durante a detecção: {e}")
                    st.stop()
            
            if results and len(results):
                annotated_frame = results[0].plot()
                annotated_image = Image.fromarray(annotated_frame)
                annotated_image_placeholder.image(annotated_image, caption='Cáries Detectadas', use_column_width=True)
            else:
                st.warning("Nenhuma cárie detectada na imagem.")
        
        submitted = st.form_submit_button("Salvar Prontuário e Detecção")
        
        if submitted:
            if not nome.strip():
                st.error("O campo 'Nome' é obrigatório.")
            else:
                prontuario_index = add_prontuario_to_faiss(
                    nome, idade, email, telefone, motivo_consulta, outras_doencas, medicamentos
                )
                st.success(f"Prontuário salvo com sucesso! ID no banco vetorial: {prontuario_index}")
                
                if annotated_image:
                    image_index = add_image_to_faiss(annotated_image, uploaded_file.name, prontuario_index)
                    st.success("Imagem anotada salva no banco vetorial.")
                elif uploaded_file is not None:
                    st.warning("Nenhuma cárie detectada na imagem. A imagem não será salva no banco vetorial.")
                else:
                    st.info("Nenhuma imagem foi enviada.")
    
    if st.button("Exibir Prontuários Armazenados"):
        st.write(prontuarios_data)

# --- Aba 2: Sistema RAG ---
with tabs[1]:
    
    
    query = st.text_input("Faça uma pergunta ao sistema:")
    
    if st.button("Buscar"):
        if query:
            keywords_image = ["imagem", "foto", "detecção", "cárie", "carie", "exame", "radiografia"]
            send_image = any(keyword in query.lower() for keyword in keywords_image)
            
            embedding = text_embedding_model.encode([query], convert_to_numpy=True)
            embedding = np.array(embedding).astype('float32')
            faiss.normalize_L2(embedding)
            I = search_faiss(embedding, index_text, k=5)
            
            context = ""
            image_path = None
            
            for idx in I:
                prontuario = prontuarios_data.get(idx, None)
                if prontuario:
                    context += f"{prontuario['prontuario_text']}\n\n"
                    if send_image and prontuario.get("imagens"):
                        image_path = prontuario["imagens"][0]
                        break
            
            if context:
                if send_image and image_path:
                    prompt = "Analise a imagem odontológica para identificar possíveis cáries ou anomalias."
                    resposta = get_multimodal_response(prompt, image_path=image_path)
                else:
                    resposta = get_text_response(query, context)
                
                st.write("**Resposta do Sistema:**")
                st.write(resposta)
                
                if send_image and image_path:
                    st.write("**Imagem Analisada:**")
                    analyzed_image = Image.open(image_path)
                    st.image(analyzed_image, caption="Imagem analisada pelo modelo", use_column_width=True)
            else:
                st.write("Nenhum prontuário relevante encontrado para a sua consulta.")
        else:
            st.warning("Por favor, insira uma pergunta para buscar.")
