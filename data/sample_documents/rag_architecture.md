# Architecture et Principes des Systèmes RAG (Retrieval-Augmented Generation)

## Introduction
Le RAG (Retrieval-Augmented Generation) est un paradigme d'intelligence artificielle combinant la recherche d'informations sémantique (Information Retrieval) et la génération de texte par des modèles de langage avancés (Large Language Models - LLM). Il permet de pallier les deux limites majeures des LLMs traditionnels : les hallucinations factuelles et le manque de connaissances privées ou récentes.

## Les Composants Clés de RAG-DocAssist

### 1. Ingestion Documentaire et Extraction
La première étape consiste à extraire le contenu brut depuis divers formats documentaires hétérogènes (fichiers PDF, documents Word .docx, fichiers texte .txt et Markdown .md). Cette étape préserve les métadonnées essentielles telles que le titre du fichier, le numéro de page d'origine et la date de versement.

### 2. Découpage Vectoriel (Chunking Stratégique)
Les documents longs ne peuvent pas être indexés d'un seul bloc. Nous utilisons un algorithme récursif (`RecursiveCharacterTextSplitter`) qui segmente le texte en morceaux de taille maîtrisée (ex: 600 caractères) avec une zone de recouvrement (*overlap* de 120 caractères). Ce recouvrement évite de briser la continuité sémantique des phrases ou des concepts situés aux frontières de découpage.

### 3. Modèles d'Embeddings Sémantiques
Chaque segment textuel est projeté dans un espace vectoriel dense à haute dimension (typiquement 384 dimensions pour `all-MiniLM-L6-v2`). La proximité géométrique (distance euclidienne ou similarité cosinus) entre deux vecteurs reflète directement leur parenté sémantique, indépendamment des mots-clés exacts utilisés.

### 4. Indexation et Moteur FAISS
L'index FAISS (Facebook AI Similarity Search) permet d'effectuer des recherches de plus proches voisins (*k-Nearest Neighbors*) en un temps inférieur à quelques millisecondes, même sur des dizaines de milliers de vecteurs. L'index peut être persisté sur disque et rechargé instantanément.

### 5. Synthèse et Attribution de Sources
Lorsqu'un utilisateur pose une question, les segments les plus proches sont insérés dans un prompt d'ancrage avec instruction stricte pour le LLM de citer explicitement ses sources. Ainsi, chaque assertion dans la réponse peut être vérifiée directement dans le document source d'origine.
