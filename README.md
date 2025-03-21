# O Sorriso em Jogo

## Índice

- [Introdução](#introdução)
- [Explicação do Problema](#explicação-do-problema)
- [Alternativas de Solução](#alternativas-de-solução)
  - [3.1. Análise de Imagens para Identificação de Problemas Odontológicos](#31-análise-de-imagens-para-identificação-de-problemas-odontológicos)
  - [3.2. Sistema de Recuperação Aumentada por Geração (RAG)](#32-sistema-de-recuperação-aumentada-por-geração-rag)
  - [3.3. Agentificação com LangChain](#33-agentificação-com-langchain)
  - [3.4. Gamificação para Incentivo à Prevenção](#34-gamificação-para-incentivo-à-prevenção)
- [Ferramentas, Bibliotecas e Frameworks Python Utilizados](#ferramentas-bibliotecas-e-frameworks-python-utilizados)
  - [4.1. Ferramentas Atuais](#41-ferramentas-atuais)
  - [4.2. Ferramentas Planejadas](#42-ferramentas-planejadas)
- [Aplicação de Conceitos e Técnicas de Machine Learning / IA](#aplicação-de-conceitos-e-técnicas-de-machine-learning--ia)
  - [5.1. Técnicas Implementadas](#51-técnicas-implementadas)
  - [5.2. Técnicas Planejadas](#52-técnicas-planejadas)
- [Implementação e Alterações na Segunda Entrega](#implementação-e-alterações-na-segunda-entrega)
  - [6.1. Alterações na Implementação](#61-alterações-na-implementação)
  - [6.2. Funcionalidades Implementadas na Versão Beta](#62-funcionalidades-implementadas-na-versão-beta)
  - [6.3. Funcionalidades Planejadas para Futuras Versões](#63-funcionalidades-planejadas-para-futuras-versões)
- [Demonstração da Versão Beta](#demonstração-da-versão-beta)
  - [7.1. Link do Vídeo de Demonstração](#71-link-do-vídeo-de-demonstração)
  - [7.2. Funcionalidades Demonstradas no Vídeo](#72-funcionalidades-demonistradas-no-vídeo)
- [Repositório do Projeto](#repositório-do-projeto)
  - [8.1. Link do Repositório](#81-link-do-repositório)

## Introdução

**"O Sorriso em Jogo"** é um aplicativo inovador voltado para a **OdontoPrev**, uma das principais operadoras de planos odontológicos do Brasil. Este aplicativo representa uma versão repaginada do aplicativo de convênio odontológico tradicional da OdontoPrev, incorporando novas funcionalidades que utilizam técnicas de Machine Learning (ML) e Inteligência Artificial (IA) generativa para um monitoramento contínuo e preventivo da saúde bucal dos usuários.

Atualmente, a entrega apresentada neste repositório é uma demonstração das funcionalidades principais implementadas utilizando **Streamlit**. Essa versão serve como um protótipo para validar conceitos e funcionalidades antes da implementação completa no ambiente móvel. O desenvolvimento completo do aplicativo está em andamento utilizando **Android Studio**, visando oferecer uma experiência nativa e otimizada para dispositivos Android.

O projeto permite que os usuários enviem fotos periódicas de sua boca e dentes, criando um hábito de monitoramento que facilita a identificação precoce de problemas odontológicos. Além disso, o aplicativo fornece insights avançados para dentistas e gestores de risco, contribuindo para a sustentabilidade das operadoras de planos odontológicos, como a OdontoPrev, ao melhorar o engajamento dos pacientes e otimizar a gestão de riscos.

## Explicação do Problema

A saúde bucal é frequentemente negligenciada até que problemas sérios, como cáries e inflamações, se manifestem. Essa negligência dificulta a identificação e o tratamento precoce dessas condições, resultando em complicações mais graves e tratamentos mais custosos. Paralelamente, operadoras de planos odontológicos enfrentam desafios como alta sinistralidade e dificuldade em prever o cancelamento de contratos (churn), o que afeta a sustentabilidade do negócio. Há uma necessidade urgente de soluções que promovam a prevenção, melhorem o engajamento dos pacientes e forneçam insights valiosos para profissionais de saúde e gestores.

## Alternativas de Solução

### 3.1. Análise de Imagens para Identificação de Problemas Odontológicos

Utilização de modelos de ML para analisar fotos enviadas pelos usuários, identificando anomalias e problemas específicos como cáries e inflamações.

### 3.2. Sistema de Recuperação Aumentada por Geração (RAG)

Criação de um sistema que extrai informações externas relevantes dos pacientes, armazenadas em um banco vetorial através de embeddings, permitindo consultas e geração de relatórios personalizados por meio de modelos de linguagem como GPT ou LLAMA 8B.

### 3.3. Agentificação com LangChain

Utilização do LangChain para automatizar a geração de relatórios e análises periódicas, sem a necessidade de intervenção humana, beneficiando tanto dentistas quanto gestores de risco.

### 3.4. Gamificação para Incentivo à Prevenção

Incorporação de elementos de gamificação para incentivar os usuários a manter hábitos saudáveis e engajar-se continuamente com o aplicativo.

## Ferramentas, Bibliotecas e Frameworks Python Utilizados

### 4.1. Ferramentas Atuais

- **Streamlit**: Construção da interface interativa, permitindo o envio de imagens, visualização de prontuários e interação com o sistema RAG.
- **PIL (Pillow)**: Manipulação de imagens enviadas pelos usuários, incluindo carregamento e exibição.
- **Ultralytics YOLO**: Modelo para detecção de cáries nas imagens enviadas.
- **FAISS (Facebook AI Similarity Search)**: Criação de índices vetoriais para dados textuais e de imagens, facilitando buscas semânticas rápidas.
- **SentenceTransformers**: Geração de embeddings para dados textuais e visuais, utilizando modelos como `all-MiniLM-L6-v2` para texto e `clip-ViT-B-32` para imagens.
- **OpenAI API (ChatGPT)**: Geração de respostas contextuais para consultas no sistema RAG, proporcionando interações baseadas em linguagem natural.

### 4.2. Ferramentas Planejadas

- **Cross-Encoder**: Reavaliação da relevância dos resultados de busca.
- **LangChain**: Automação de processos e geração de relatórios periódicos.
- **AutoEncoders**: Identificação de anomalias em imagens de forma temporal.

## Aplicação de Conceitos e Técnicas de Machine Learning / IA

### 5.1. Técnicas Implementadas

- **YOLO para Detecção de Cáries**: Utilização do modelo YOLO para identificar áreas de possível anomalia nas imagens enviadas, destacando cáries e inflamações.
- **Geração de Embeddings e Busca Semântica**: Uso do SentenceTransformers para gerar embeddings das informações dos pacientes e do FAISS para armazenar e facilitar buscas semânticas eficientes.
- **Respostas Contextualizadas com ChatGPT**: Integração com a API do ChatGPT para gerar respostas baseadas no contexto e nos dados dos prontuários armazenados.

### 5.2. Técnicas Planejadas

- **Reordenação de Resultados com Cross-Encoder**: Implementação futura para melhorar a relevância das respostas por meio de reordenação dos resultados de busca.
- **Automação com LangChain**: Integração para geração automática de relatórios e análises periódicas.
- **Detecção de Anomalias com AutoEncoders**: Análise de variações nas imagens odontológicas ao longo do tempo para identificar anomalias de maneira mais eficaz.

## Implementação e Alterações na Segunda Entrega

### 6.1. Alterações na Implementação

Comparando com a proposta original, algumas adaptações foram realizadas para otimizar a funcionalidade e a performance da versão Beta:

- **Detecção de Cáries com YOLO**: Substituição da arquitetura CNN (ResNet) pelo modelo YOLO para aumentar a precisão e eficiência da detecção em tempo real.
- **Processamento de Imagem com PIL**: Adoção da biblioteca PIL (Pillow) para manipulação de imagens, simplificando a integração com o Streamlit e eliminando a necessidade do OpenCV.

### 6.2. Funcionalidades Implementadas na Versão Beta

- **Prontuário Odontológico e Detecção de Cáries**: Interface para entrada de dados do paciente e envio de imagens, com análise de cáries utilizando YOLO.
- **Armazenamento e Busca Semântica com FAISS**: Conversão das informações dos pacientes e imagens em embeddings, indexadas no FAISS para buscas eficientes.
- **Sistema RAG com ChatGPT**: Permite consultas no sistema RAG e recebe respostas contextuais baseadas nos dados armazenados.

### 6.3. Funcionalidades Planejadas para Futuras Versões

- **Reordenação de Resultados com Cross-Encoder**: Melhorar a relevância das respostas através de reordenação dos resultados de busca.
- **Automação de Relatórios com LangChain**: Geração automática de relatórios e análises periódicas para acompanhamento odontológico.
- **Análise Temporal com AutoEncoders**: Detectar variações nas imagens ao longo do tempo para identificar mudanças e anomalias na saúde bucal dos usuários.

## Demonstração da Versão Beta

### 7.1. Link do Vídeo de Demonstração

[https://youtu.be/wECOeFML8kM](#)

### 7.2. Funcionalidades Demonstradas no Vídeo

- **Envio de Prontuário**: Interface para entrada de dados do paciente e envio de imagens.
- **Detecção de Cáries com YOLO**: Análise das imagens enviadas e exibição das detecções de cáries.
- **Busca Semântica e Respostas Contextuais**: Demonstração da busca de informações e geração de respostas através do sistema RAG.

## Repositório do Projeto

### 8.1. Link do Repositório

[https://github.com/rafaelnxd/IotSprint03](#)
