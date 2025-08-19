from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import json
import hashlib
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
import torch
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import functools
from .cache_utils import memory_cache, disk_cache

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize models on first use
_SUMMARIZER = None
_EXPLANATION_MODEL = None
_EXPLANATION_TOKENIZER = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_summarizer():
    """Get a cached instance of the summarization pipeline."""
    global _SUMMARIZER
    if _SUMMARIZER is None:
        _SUMMARIZER = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if _DEVICE == "cuda" else -1,
            torch_dtype=torch.float16 if _DEVICE == "cuda" else torch.float32
        )
    return _SUMMARIZER

def get_explanation_model():
    """Get a cached instance of the explanation model."""
    global _EXPLANATION_MODEL, _EXPLANATION_TOKENIZER
    if _EXPLANATION_MODEL is None or _EXPLANATION_TOKENIZER is None:
        model_name = "google/flan-t5-base"
        _EXPLANATION_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _EXPLANATION_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if _DEVICE == "cuda" else torch.float32,
            device_map="auto" if _DEVICE == "cuda" else None,
            low_cpu_mem_usage=True
        )
    return _EXPLANATION_MODEL, _EXPLANATION_TOKENIZER

@disk_cache(ttl=86400)  # Cache for 24 hours
def _get_paper_summary_impl(paper_text: str, max_length: int = 150) -> str:
    """Internal implementation of get_paper_summary with caching."""
    try:
        # First try to use the abstract if available
        if 'abstract' in paper_text.lower():
            abstract_start = paper_text.lower().find('abstract')
            if abstract_start != -1:
                abstract = paper_text[abstract_start:abstract_start + 500]  # Get first 500 chars of abstract
                return abstract.strip()
        
        # Fall back to summarization if no abstract found
        summarizer = get_summarizer()
        
        # Use a more efficient truncation strategy
        max_model_length = 1024
        paper_text = paper_text[:max_model_length]  # Simple truncation first
        
        # Split into sentences and take first few if too long
        sentences = sent_tokenize(paper_text)
        if len(sentences) > 10:  # If more than 10 sentences, take first 5 and last 2
            paper_text = ' '.join(sentences[:5] + sentences[-2:])
        
        try:
            summary = summarizer(
                paper_text,
                max_length=max_length,
                min_length=30,
                do_sample=False,
                truncation=True
            )
            return summary[0]['summary_text']
        except Exception as e:
            # Fallback to first few sentences
            return ' '.join(sentences[:3]) if sentences else "No summary available."
    except Exception as e:
        return "Summary not available."

def get_paper_summary(paper: Dict[str, Any], max_length: int = 250) -> str:
    """Generate a summary of a research paper with caching."""
    paper_text = f"{paper.get('title', '')}. {paper.get('summary', '')}"
    return _get_paper_summary_impl(paper_text, max_length)

@disk_cache(ttl=86400)  # Cache for 24 hours
def _get_concept_explanation_impl(concept: str, context_text: str = "") -> str:
    """Internal implementation of get_concept_explanation with caching."""
    try:
        model, tokenizer = get_explanation_model()
        
        # Prepare the prompt - keep it concise
        if context_text:
            # Truncate context to avoid excessive input length
            max_context_length = 512
            if len(context_text) > max_context_length:
                context_text = context_text[:max_context_length] + "..."
                
            prompt = f"""Explain in simple terms: {concept}
            
            Context from research papers:
            {context_text}
            
            Explanation: """
        else:
            prompt = f"Explain in simple terms: {concept} - "
        
        # Generate explanation with optimized settings
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if _DEVICE == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,  # Shorter for faster generation
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        # Clean up the output
        explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        explanation = explanation.replace(prompt, "").strip()
        return explanation if explanation else "Explanation not available."
    except Exception as e:
        print(f"Error generating explanation: {str(e)}")
        return "Explanation not available at this time."

def get_concept_explanation(concept: str, context_papers: List[Dict[str, Any]] = None) -> str:
    """Generate an explanation of a concept using the LLM with caching."""
    # Prepare context text for caching
    context_text = ""
    if context_papers and len(context_papers) > 0:
        context_text = "\n".join([f"Title: {p.get('title', '')}\nSummary: {p.get('summary', '')}" 
                                 for p in context_papers[:3]])  # Use top 3 papers for context
    
    return _get_concept_explanation_impl(concept, context_text)

def extract_key_concepts(text: str, top_n: int = 10) -> List[str]:
    """Extract key concepts from text using simple frequency-based approach."""
    # Tokenize and clean text
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Count word frequencies
    freq_dist = nltk.FreqDist(words)
    
    # Get most common words (excluding very common ones)
    common_words = {'research', 'paper', 'study', 'result', 'method', 'model', 'data', 'result', 'show'}
    key_concepts = [word for word, _ in freq_dist.most_common(top_n * 2) 
                   if word not in common_words and len(word) > 2][:top_n]
    
    return key_concepts

@memory_cache(maxsize=10)  # Cache in memory for the session
def _get_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings for a list of texts with caching."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-mpnet-base-v2')
    return model.encode(texts, show_progress_bar=False)

@disk_cache(ttl=3600)  # Cache for 1 hour
def _get_pca_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Apply PCA to reduce embeddings to 2D with caching."""
    pca = PCA(n_components=2)
    return pca.fit_transform(embeddings)

def visualize_concept_relationships(queries: List[str], papers: List[Dict[str, Any]]) -> go.Figure:
    """Create a visualization of concept relationships using PCA with caching."""
    if not queries or not papers:
        return go.Figure()
    
    # Create a unique key for caching
    cache_key = {
        'queries': queries,
        'paper_ids': [p.get('id', '') for p in papers[:50]],  # Limit to 50 papers for performance
        'paper_titles': [p.get('title', '') for p in papers[:50]]
    }
    
    # Create a unique hash for the cache key
    cache_key_str = hashlib.md5(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()
    
    # Extract text for embedding
    texts = queries.copy()
    paper_texts = []
    for paper in papers[:50]:  # Limit number of papers for performance
        paper_text = f"{paper.get('title', '')} {paper.get('summary', '')}"
        paper_texts.append(paper_text)
        texts.append(paper_text)
    
    # Get embeddings with caching
    embeddings = _get_embeddings(texts)
    
    # Get PCA-reduced embeddings with caching
    embeddings_2d = _get_pca_embeddings(embeddings)
    
    # Split back into queries and papers
    query_embeddings = embeddings_2d[:len(queries)]
    paper_embeddings = embeddings_2d[len(queries):]
    
    # Create figure
    fig = go.Figure()
    
    # Add paper points
    fig.add_trace(go.Scatter(
        x=paper_embeddings[:, 0],
        y=paper_embeddings[:, 1],
        mode='markers',
        name='Papers',
        text=[p.get('title', '') for p in papers[:50]],
        hoverinfo='text',
        marker=dict(size=8, opacity=0.6)
    ))
    
    # Add query points
    fig.add_trace(go.Scatter(
        x=query_embeddings[:, 0],
        y=query_embeddings[:, 1],
        mode='markers+text',
        name='Queries',
        text=queries,
        textposition='top center',
        marker=dict(size=12, color='red')
    ))
    
    # Update layout
    fig.update_layout(
        title='Concept Space Visualization',
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        showlegend=True,
        hovermode='closest'
    )
    
    return fig
