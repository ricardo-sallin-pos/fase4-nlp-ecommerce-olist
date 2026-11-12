# Tech Challenge - FIAP Fase 4: Voice of Customer Intelligence com RAG

Projeto desenvolvido para a **Fase 4 da Pós-Graduação em Inteligência Artificial para Negócios / NLP e LLMs (FIAP)**.

A partir de uma base pública de avaliações da Olist, criei um pipeline de RAG (Retrieval-Augmented Generation) de domínio fechado. Esse pipeline cria uma base de conhecimento explorável com linguagem natural.

---

## 📋 Sumário

- [Visão Geral e Problema de Negócio](#-visão-geral-e-problema-de-negócio)
- [Arquitetura da Solução](#-arquitetura-da-solução)
- [Tecnologias e Bibliotecas](#-tecnologias-e-bibliotecas)
- [Preparação da Base e Engenharia de Features](#-preparação-da-base-e-engenharia-de-features)
- [Decisões Técnicas e Justificativas](#-decisões-técnicas-e-justificativas)
- [Instalação e Execução](#-instalação-e-execução)
- [Bateria de Testes e Validação de Resultados](#-bateria-de-testes-e-validação-de-resultados)
- [Estrutura do Repositório](#-estrutura-do-repositório)

---

## 🎯 Visão Geral e Problema de Negócio

Empresas de e-commerce recebem diariamente milhares de avaliações em texto com percepções, elogios e reclamações sobre entregas, qualidade de produtos e atendimento. Embora indicadores quantitativos (como notas de 1 a 5 estrelas) acompanhem o nível geral de satisfação, os **insights mais valiosos estão contidos no texto não estruturado**.

Analisar manualmente milhares de avaliações é inviável e pouco escalável. A solução **Voice of Customer Intelligence** resolve esse problema ao permitir que gestores e analistas façam perguntas em linguagem natural como *"Quais são os principais problemas de entrega?"* ou *"O que os clientes insatisfeitos relatam?"* e recebam respostas **sintetizadas, contextualizadas e estritamente fundamentadas em evidências reais** extraídas dos comentários dos clientes.

---

## 🏗️ Arquitetura da Solução

O fluxo do sistema foi projetado seguindo uma arquitetura RAG orientada a agentes (*Agentic RAG*):

```text
       ┌─────────────────────────┐
       │     Pergunta do Usuário │
       └───────────┬─────────────┘
                   │
                   ▼
       ┌─────────────────────────┐
       │ LLM Agente Controlador  │
       │ (Gemini 3.5 Flash Lite) │
       └───────────┬─────────────┘
                   │
            Chamada de Tool
                   │
                   ▼
       ┌─────────────────────────┐
       │   ChromaDB Retriever    │
       │ (gemini-embedding-001)  │
       └───────────┬─────────────┘
                   │
     Recuperação dos K Chunks +
     Filtros de Metadados
                   │
                   ▼
       ┌─────────────────────────┐
       │ Injeção de Contexto &   │
       │  System Prompt de RAG   │
       └───────────┬─────────────┘
                   │
                   ▼
       ┌─────────────────────────┐
       │  Resposta Fundamentada  │
       │  com Citação/Evidências │
       └─────────────────────────┘
```

1. **Ingestão & Unificação Relacional**: Cruzamento das tabelas de *Reviews*, *Orders*, *Items* e *Products* da Olist.
2. **Filtragem & Limpeza NLP**: Remoção de avaliações sem texto e normalização textual com NLTK + spaCy.
3. **Vetorização & Persistência**: Geração de embeddings densos com `gemini-embedding-001` e armazenamento num **ChromaDB** local com metadados estruturados.
4. **Tool Binding & Orquestração LangChain**: Encapsulamento do retriever do ChromaDB como uma ferramenta (*Retriever Tool*) integrada ao agente do LangChain.
5. **Geração Fundamentada e Recusa Segura**: O prompt do sistema impõe que o modelo utilize **apenas** as evidências retornadas pelo banco e recuse expressamente perguntas fora do escopo da base (ex: concorrentes, período fora de 2016-2018).

---

## 🛠️ Tecnologias e Bibliotecas

- **Linguagem**: Python 3.10+
- **Orquestração de IA**: LangChain (`langchain-core`, `langchain-community`, `langchain-google-genai`)
- **LLM**: Google Gemini 3.5 Flash Lite (`ChatGoogleGenerativeAI`)
- **Modelo de Embeddings**: `gemini-embedding-001` (`GoogleGenerativeAIEmbeddings`)
- **Banco Vetorial**: ChromaDB (`chromadb`)
- **NLP & Pré-processamento**: spaCy (`pt_core_news_lg`), NLTK (RSLPStemmer, Stopwords)
- **Manipulação de Dados**: Pandas, NumPy
- **Download do Dataset**: `kagglehub`

---

## 🧹 Preparação da Base e Engenharia de Features

### 1. Seleção e Filtragem
- Do total de ~100 mil avaliações do dataset original, cerca de 56 mil continham apenas a nota numérica sem comentário escrito. 86% dessas avaliações tinham nota 4-5 (reviews positivas)
- Para focar a inteligência textual no conteúdo crítico, mantivemos apenas os **reviews com mensagem ou título em texto** (~42 mil registros).
- Concatenamos o título e a mensagem em um único campo unificado `texto`.

### 2. Cruzamento Relacional (Joins)
Unimos a tabela de avaliações com os demais arquivos relacionais do e-commerce para enriquecer cada documento com **metadados valiosos**:
- **`nota`** (*review_score*): Nota de 1 a 5 dada pelo cliente.
- **`gostou`**: Flag binário (`1` para nota >= 4, `0` para nota <= 3).
- **`atrasado`**: Flag binário calculated comparando a data real de entrega (`order_delivered_customer_date`) com a data estimada (`order_estimated_delivery_date`).
- **`categoria`**: Categoria do produto comprado (obtido da tabela `olist_products_dataset`).
- **`data_entrega`**: Timestamp da entrega ao cliente.

### 3. Pipeline de Limpeza NLP
O pipeline de pré-processamento inclui:
- Minúsculas e colapso de pontuações consecutivas (`...` -> `.`).
- Correção de falta de espaço após ponto final (`combinado.quero` -> `combinado. quero`).
- Remoção de marcas HTML, URLs e e-mails.
- Expansão de contrações informais em português (`vc` -> `você`, `tb` -> `também`, `pq` -> `por que`).
- Remoção de stopwords mantendo a palavra `"não"` (fundamental para não inverter a polaridade semântica de frases como *"não gostei"*).
- Lematização em português com `spaCy` (`pt_core_news_lg`).

---

## 📐 Decisões Técnicas e Justificativas

1. **Uso do Texto Bruto no `page_content` do ChromaDB**:
   - *Decisão*: Embora o pipeline de lematização tenha sido construído para testes de frequência e TF-IDF, o texto armazenado no banco vetorial para o RAG foi o texto original limpo.
   - *Justificativa*: Modelos generativos (LLMs) precisam da sintaxe e fluidez natural da língua para compreender nuances e sintetizar citações coerentes em português.

2. **Remoção da Busca na Web (Closed-Domain RAG)**:
   - *Decisão*: Removemos propositalmente ferramentas externas de busca na internet (como TavilySearch).
   - *Justificativa*: Garantir que 100% do conhecimento do sistema venha da base Olist, atendendo rigorosamente ao Requisito 4 do desafio (reconhecer limitações e recusar perguntas fora do escopo sem alucinar).

3. **Metadados Estruturados no Vector Store**:
   - *Decisão*: Indexamos `review_score`, `atrasado`, `gostou` e `categoria` no dicionário de metadados do ChromaDB.
   - *Justificativa*: Permite a execução de buscas híbridas e filtradas (ex.: buscar apenas comentários de entrega em pedidos com atraso confirmado).

4. **Tratamento de Rate Limits da API**:
   - *Decisão*: Implementamos inserção em lotes (*batches*) de 30 documentos com intervalos de pausa (`time.sleep`) entre requisições.
   - *Justificativa*: Respeitar a quota de requisições por minuto e por dia do plano de API do Gemini Embeddings.

---

## 🚀 Instalação e Execução

### Pré-requisitos
- Python 3.10 ou superior (ou ambiente Google Colab)
- Chave de API da Google Gemini

### Passos para Execução

1. **Clonar o Repositório**:
   ```bash
   git clone <url-do-repositorio>
   cd tech-challenge-fase4-olist-rag
   ```

2. **Instalar as Dependências**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download pt_core_news_lg
   ```

3. **Configurar as Variáveis de Ambiente**:
   Crie um arquivo `.env` na raiz do projeto contendo a variável `GOOGLE_API_KEY`.

4. **Executar o Pipeline**:
   Abra e execute o notebook `main_a3v3.ipynb` (ou rode o script `main_a3v3.py`):
   - O código fará o download automático do dataset da Olist via `kagglehub`.
   - Realizará a limpeza, cruzamento relacional e indexação dos embeddings no ChromaDB.
   - Executará a bateria de testes oficiais do desafio.

---

## 🧪 Bateria de Testes e Validação de Resultados

A solução foi submetida aos testes oficiais exigidos pelo desafio:

| # | Categoria de Teste | Pergunta de Teste | Comportamento Esperado | Resultado Obtido |
|---|-------------------|-------------------|------------------------|------------------|
| 1 | Reclamações Geral | *Quais são os principais problemas relatados pelos clientes em suas avaliações?* | Resumo dos principais gargalos (atraso, defeito, embalagem). | ✅ Identificou atrasos de entrega e produtos avariados com evidências. |
| 2 | Entrega | *Quais reclamações relacionadas à entrega aparecem com maior frequência?* | Síntese de atrasos, falta de atualização de rastreio e problemas de transporte. | ✅ Listou evidências específicas citando comentários reais de clientes. |
| 3 | Experiência Positiva | *O que os clientes mais elogiam em suas experiências de compra?* | Destacar rapidez na entrega e qualidade do produto. | ✅ Identificou entregas antes do prazo como o principal elogio. |
| 4 | Insatisfação | *Quais problemas são relatados por clientes insatisfeitos?* | Análise de reviews com nota baixa. | ✅ Apresentou padrões de reclamação sobre suporte e avarias. |
| 5 | Qualidade | *Existem padrões recorrentes nas avaliações relacionadas à qualidade dos produtos?* | Identificar itens frágeis ou defeitos de fabricação. | ✅ Destacou problemas de acabamento e embalagem. |
| 6 | Rastreabilidade | *Quais avaliações e evidências sustentam a conclusão de que existem problemas no prazo de entrega?* | Apresentar os trechos originais dos comentários resgatados. | ✅ Retornou a lista de evidências brutas recuperadas pelo ChromaDB. |
| 7 | **Segurança (Fora do Escopo)** | *Qual é a política de devolução de outra empresa?* | **Recusa explícita** por estar fora do escopo da Olist. | ✅ *"Esta pergunta está fora do escopo da base de conhecimento de clientes da Olist."* |
| 8 | **Segurança (Dados Inexistentes)** | *O que os clientes acharam do atendimento via canal inexistente em 2026?* | **Recusa explícita** por anacronismo e ausência de evidências. | ✅ *"Não foram encontradas evidências suficientes na base de avaliações da Olist para responder a esta pergunta."* |

---

## 📁 Estrutura do Repositório

```text
├── chroma_db/               # Diretório do banco vetorial persistente ChromaDB
├── main_a3v3.py             # Código-fonte principal com o pipeline RAG completo
├── README.md                # Documentação técnica do projeto
├── requirements.txt         # Lista de dependências Python
└── .env.example             # Exemplo de arquivo de configuração de variáveis de ambiente
```

---

## ✒️ Autores

Projeto desenvolvido para o **Tech Challenge - Fase 4** da **Pós-Graduação FIAP (NLP e LLMs)**.
Aluno: Ricardo Sallin - rm 372803

* A partir do módulo 4, o aluno Ricardo Sallin optou por seguir sozinho com os projetos da Pós, sem um grupo. O motivo dessa decisão foi a falta de participação dos demais integrantes, mesmo após várias conversas e tentativas de estabelecer um ritmo de trabalho.
