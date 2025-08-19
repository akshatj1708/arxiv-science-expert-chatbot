import os
import json
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
from tqdm import tqdm
from typing import List, Dict, Any, Union, Optional, Tuple, Callable
import arxiv
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import functools
import hashlib

# Import caching utilities
from .cache_utils import disk_cache, memory_cache

class SearchOperator(Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    PHRASE = "PHRASE"

@dataclass
class SearchTerm:
    term: str
    operator: SearchOperator
    field: str = "all"  # all, title, abstract, author, category
    
    def __str__(self):
        return f"{self.operator.value} {self.field}:{self.term}" if self.field != "all" else f"{self.operator.value} {self.term}"

# Cache for the sentence transformer model
@memory_cache(maxsize=1)
def get_sentence_transformer(model_name: str = 'all-mpnet-base-v2') -> SentenceTransformer:
    """Get a cached instance of the sentence transformer model."""
    return SentenceTransformer(model_name)

# Initialize the model
model = get_sentence_transformer()

class ArxivDataset:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.papers = []
        self.index = None
        self.embeddings = None
        self.keyword_index = {
            'title': {},
            'authors': {},
            'categories': {},
            'year': {}
        }
        self.vectorizer = None
        self.tfidf_matrix = None
        
    @disk_cache(ttl=86400)  # Cache for 24 hours
    def _download_arxiv_data_impl(self, category: str, max_results: int = 100) -> List[Dict]:
        """Internal implementation of download_arxiv_data with caching."""
        try:
            client = arxiv.Client(
                page_size=100,  # Fetch 100 papers per request
                delay_seconds=3,  # Add delay between requests
                num_retries=3  # Retry failed requests
            )
            
            # Search for papers in the specified category
            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=min(max_results, 1000),  # Limit to 1000 results
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            papers = []
            try:
                # Use list() to force evaluation of the generator
                results = list(client.results(search))
                if not results:
                    print(f"No results found for category: {category}")
                    return []
                    
                for result in tqdm(results[:max_results], desc=f"Fetching {category} papers"):
                    try:
                        paper = {
                            'id': result.entry_id.split('/')[-1] if result.entry_id else f"unknown_{len(papers)}",
                            'title': result.title or "No title",
                            'authors': [author.name for author in result.authors] if hasattr(result, 'authors') else [],
                            'published': result.published.strftime('%Y-%m-%d') if hasattr(result, 'published') and result.published else None,
                            'updated': result.updated.strftime('%Y-%m-%d') if hasattr(result, 'updated') and result.updated else None,
                            'summary': result.summary if hasattr(result, 'summary') else "",
                            'categories': result.categories if hasattr(result, 'categories') else [category],
                            'doi': getattr(result, 'doi', None),
                            'pdf_url': getattr(result, 'pdf_url', None),
                            'comment': getattr(result, 'comment', None),
                            'journal_ref': getattr(result, 'journal_ref', None),
                            'primary_category': getattr(result, 'primary_category', category),
                            'links': [link.href for link in result.links] if hasattr(result, 'links') else [],
                            'pdf': None  # Will store the PDF content if downloaded
                        }
                        papers.append(paper)
                    except Exception as e:
                        print(f"Error processing paper: {e}")
                        continue
                        
                if not papers:
                    print(f"No valid papers found for category: {category}")
                    return []
                    
                return papers
                
            except Exception as e:
                print(f"Error fetching papers: {e}")
                return []
                
        except Exception as e:
            print(f"Error initializing arXiv client: {e}")
            return []

    def download_arxiv_data(self, category: str, max_results: int = 100, force_download: bool = False) -> List[Dict]:
        """Download papers from arXiv for a specific category with caching."""
        try:
            # Normalize category for filename
            file_category = category.replace('.', '').lower()
            
            # Create cache key based on function name and arguments
            cache_key = f"{category}_{max_results}"
            
            if force_download:
                # Clear cache if force_download is True
                cache = getattr(self._download_arxiv_data_impl, 'cache', None)
                if cache and hasattr(cache, 'delete'):
                    cache.delete(cache_key)
            
            print(f"Attempting to download up to {max_results} papers for category: {category}")
            
            # Get data (from cache or fresh download)
            papers = self._download_arxiv_data_impl(category, max_results)
            
            if not papers:
                print(f"Warning: No papers found for category: {category}")
                # Try with a different category if the default fails
                if category == 'cs':
                    print("Trying with 'cs.CL' category instead...")
                    return self.download_arxiv_data('cs.CL', max_results, force_download)
                return []
                
            print(f"Successfully downloaded {len(papers)} papers for category: {category}")
            
            # Save to file for persistence
            os.makedirs(self.data_dir, exist_ok=True)
            file_path = os.path.join(self.data_dir, f"{file_category}_papers.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(papers, f, ensure_ascii=False, indent=2)
            
            self.papers = papers
            return papers
            
        except Exception as e:
            print(f"Error in download_arxiv_data: {str(e)}")
            return []
    
    def load_arxiv_data(self, category: str, force_download: bool = False) -> List[Dict[str, Any]]:
        """Load papers from file or download if not exists."""
        # Normalize category for filename (same as in download_arxiv_data)
        file_category = category.replace('.', '').lower()
        file_path = os.path.join(self.data_dir, f"{file_category}_papers.json")
        
        try:
            if force_download or not os.path.exists(file_path):
                print(f"Downloading papers for category: {category}")
                self.papers = self.download_arxiv_data(category, max_results=50)  # Reduced to 50 for testing
            else:
                print(f"Loading papers from cache: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.papers = json.load(f)
            
            if not self.papers:
                print(f"No papers found for category: {category}")
                return []
                
            print(f"Successfully loaded {len(self.papers)} papers for category: {category}")
            
            # Create embeddings for the papers if we have any
            try:
                self._create_embeddings()
            except Exception as e:
                print(f"Error creating embeddings: {e}")
                # Continue without embeddings rather than failing
                
            return self.papers
            
        except Exception as e:
            print(f"Error in load_arxiv_data: {str(e)}")
            return []
    
    @memory_cache(maxsize=1)
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts with memory caching."""
        return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    def _create_embeddings(self):
        """Create embeddings and indices for paper content with caching."""
        if not self.papers:
            raise ValueError("No papers loaded. Call load_arxiv_data() first.")
            
        # Create a unique cache key for these papers
        paper_ids = [p['id'] for p in self.papers]
        cache_key = hashlib.md5(''.join(sorted(paper_ids)).encode('utf-8')).hexdigest()
        
        # Create embeddings for paper titles and abstracts
        texts = [f"{p['title']} {p['summary']}" for p in self.papers]
        
        # Use cached embeddings if available, otherwise create new ones
        self.embeddings = self._get_embeddings(tuple(texts))  # Convert to tuple for hashing
        
        # Create FAISS index for efficient similarity search
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        # Build keyword indices
        self._build_keyword_indices()
        
    def _build_keyword_indices(self):
        """Build keyword indices for fast filtering."""
        for idx, paper in enumerate(self.papers):
            # Index by title words
            for word in re.findall(r'\w+', paper.get('title', '').lower()):
                if word not in self.keyword_index['title']:
                    self.keyword_index['title'][word] = set()
                self.keyword_index['title'][word].add(idx)
            
            # Index by authors
            for author in paper.get('authors', []):
                if author not in self.keyword_index['authors']:
                    self.keyword_index['authors'][author] = set()
                self.keyword_index['authors'][author].add(idx)
            
            # Index by categories
            for category in paper.get('categories', []):
                if category not in self.keyword_index['categories']:
                    self.keyword_index['categories'][category] = set()
                self.keyword_index['categories'][category].add(idx)
            
            # Index by year
            if 'published' in paper:
                try:
                    year = paper['published'][:4]
                    if year not in self.keyword_index['year']:
                        self.keyword_index['year'][year] = set()
                    self.keyword_index['year'][year].add(idx)
                except (ValueError, IndexError):
                    pass
    
    def _parse_search_query(self, query: str) -> List[SearchTerm]:
        """Parse a search query into search terms with operators."""
        if not query.strip():
            return []
            
        terms = []
        # Handle phrase searches (text in quotes)
        phrases = re.findall(r'"(.*?)"', query)
        for phrase in phrases:
            query = query.replace(f'"{phrase}"', '')
            terms.append(SearchTerm(phrase, SearchOperator.PHRASE))
        
        # Handle field-specific searches (e.g., author:smith)
        field_queries = re.findall(r'(\w+):([^\s\-]+(?:\s+[^\s\-]+)*)', query)
        for field, value in field_queries:
            query = query.replace(f'{field}:{value}', '')
            terms.append(SearchTerm(value, SearchOperator.AND, field.lower()))
        
        # Handle remaining terms with AND operator
        remaining_terms = re.split(r'\s+(?:AND|OR|NOT)\s+', query)
        for term in remaining_terms:
            if term.upper() in ['AND', 'OR', 'NOT']:
                continue
            if term.strip():
                terms.append(SearchTerm(term.strip(), SearchOperator.AND))
        
        return terms
    
    def _evaluate_search_term(self, term: SearchTerm) -> set:
        """Evaluate a single search term and return matching paper indices."""
        if term.field == 'all':
            # Search in all fields
            results = set()
            for field in ['title', 'authors', 'categories']:
                if term.term.lower() in self.keyword_index[field]:
                    results.update(self.keyword_index[field][term.term.lower()])
            return results
        elif term.field in self.keyword_index:
            # Search in specific field
            if term.operator == SearchOperator.PHRASE:
                # For phrase search, we need to check the actual text
                return set(
                    i for i, paper in enumerate(self.papers)
                    if term.term.lower() in paper.get(term.field, '').lower()
                )
            else:
                # For regular term search, use the index
                return self.keyword_index[term.field].get(term.term.lower(), set())
        return set()
    
    def search_papers(
        self, 
        query: str, 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = 'relevance',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Search for papers with advanced querying and filtering.
        
        Args:
            query: Search query string (supports boolean operators)
            k: Maximum number of results to return
            filters: Dictionary of filters (e.g., {'category': ['cs.CL', 'cs.AI']})
            sort_by: Sort method ('relevance', 'date', 'citations')
            start_date: Filter papers published after this date
            end_date: Filter papers published before this date
            
        Returns:
            Tuple of (list of matching papers, total number of matches)
        """
        if not self.index or not self.papers:
            return [], 0
        
        # Parse the query into search terms
        search_terms = self._parse_search_query(query)
        
        # Start with all papers and apply filters
        matching_indices = set(range(len(self.papers)))
        
        # Apply search terms
        if search_terms:
            term_results = []
            for term in search_terms:
                term_matches = self._evaluate_search_term(term)
                if term.operator == SearchOperator.NOT:
                    matching_indices -= term_matches
                else:
                    term_results.append(term_matches)
            
            if term_results:
                # Combine with AND logic for multiple terms
                matching_indices &= set.intersection(*term_results) if term_results else set()
        
        # Apply filters
        if filters:
            for field, values in filters.items():
                if not values:
                    continue
                if field in self.keyword_index:
                    field_matches = set()
                    for value in values:
                        if value in self.keyword_index[field]:
                            field_matches.update(self.keyword_index[field][value])
                    matching_indices &= field_matches
               
        # Apply date filters
        if start_date or end_date:
            date_filtered = set()
            for idx in matching_indices:
                paper = self.papers[idx]
                try:
                    paper_date = datetime.strptime(paper.get('published', ''), '%Y-%m-%d')
                    if start_date and paper_date < start_date:
                        continue
                    if end_date and paper_date > end_date:
                        continue
                    date_filtered.add(idx)
                except (ValueError, TypeError):
                    continue
            matching_indices = date_filtered
        
        # Convert to list and get the papers
        matching_indices = list(matching_indices)
        total_matches = len(matching_indices)
        
        # Sort results
        if sort_by == 'date' and 'published' in self.papers[0]:
            matching_indices.sort(
                key=lambda i: self.papers[i].get('published', ''), 
                reverse=True
            )
        elif sort_by == 'citations' and 'citation_count' in self.papers[0]:
            matching_indices.sort(
                key=lambda i: self.papers[i].get('citation_count', 0), 
                reverse=True
            )
        else:  # Default to relevance
            if query.strip() and search_terms:
                # Re-rank by semantic similarity for relevant searches
                query_embedding = model.encode([query])
                paper_embeddings = self.embeddings[matching_indices]
                distances = np.linalg.norm(
                    paper_embeddings - query_embedding, 
                    axis=1
                )
                # Sort by distance (ascending)
                matching_indices = [
                    idx for _, idx in sorted(zip(distances, matching_indices))
                ]
        
        # Get the top k results
        results = []
        for idx in matching_indices[:k]:
            paper = self.papers[idx].copy()
            # Add relevance score if available
            if query.strip() and search_terms:
                paper['relevance_score'] = float(1 / (1 + distances[idx]))
            results.append(paper)
        
        return results, total_matches

# Global instance
arxiv_dataset = ArxivDataset()

@disk_cache(ttl=3600)  # Cache for 1 hour
def _search_papers_impl(
    query: str, 
    papers: List[Dict[str, Any]], 
    k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    sort_by: str = 'relevance',
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """Internal implementation of search_papers with caching."""
    # Create a dataset instance if not already created
    global arxiv_dataset
    if not hasattr(arxiv_dataset, 'papers') or not arxiv_dataset.papers:
        arxiv_dataset.papers = papers
        
    # Set up filters
    if filters is None:
        filters = {}
        
    # Apply date filters if provided
    if start_date or end_date:
        filtered_papers = []
        for paper in papers:
            paper_date = datetime.strptime(paper['published'], '%Y-%m-%d') if paper.get('published') else None
            if paper_date:
                if start_date and paper_date < start_date:
                    continue
                if end_date and paper_date > end_date:
                    continue
                filtered_papers.append(paper)
        papers = filtered_papers
    
    # Apply other filters
    for field, values in filters.items():
        if field == 'category' and values:
            papers = [p for p in papers if any(cat in values for cat in p.get('categories', []))]
        elif field == 'author' and values:
            papers = [p for p in papers if any(author in p.get('authors', []) for author in values)]
        elif field in ['year', 'month'] and values:
            papers = [p for p in papers if p.get(field) in values]
    
    # If no query, return filtered results with proper sorting
    if not query.strip():
        if papers:
            if sort_by == 'date':
                papers = sorted(papers, key=lambda x: x.get('published', ''), reverse=True)
            elif sort_by == 'relevance':
                # Default to sorting by date if no query for relevance
                papers = sorted(papers, key=lambda x: x.get('published', ''), reverse=True)
        return papers[:k], len(papers)
    
    # Perform search with the dataset's search functionality
    results, total = arxiv_dataset.search_papers(
        query=query,
        k=k,
        filters=filters,
        sort_by=sort_by,
        start_date=start_date,
        end_date=end_date
    )
    
    return results, total

def search_papers(
    query: str, 
    papers: List[Dict[str, Any]], 
    k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    sort_by: str = 'relevance',
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    force_refresh: bool = False
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Search for papers with advanced querying and filtering and caching.
    
    Args:
        query: Search query string (supports boolean operators)
        papers: List of papers to search through
        k: Maximum number of results to return
        filters: Dictionary of filters (e.g., {'category': ['cs.CL', 'cs.AI']})
        sort_by: Sort method ('relevance', 'date', 'citations')
        start_date: Filter papers published after this date
        end_date: Filter papers published before this date
        force_refresh: If True, bypass cache and force a fresh search
        
    Returns:
        Tuple of (list of matching papers, total number of matches)
    """
    # Create a cache key based on the search parameters
    def make_serializable(obj):
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple, set)):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, 'isoformat'):  # Handle datetime objects
            return obj.isoformat()
        return str(obj)
    
    # Create a serializable cache key
    cache_key = {
        'query': query,
        'num_papers': len(papers),
        'k': k,
        'filters': make_serializable(filters) if filters else None,
        'sort_by': sort_by,
        'start_date': start_date.isoformat() if start_date else None,
        'end_date': end_date.isoformat() if end_date else None
    }
    
    # Convert to a hashable key
    cache_key_str = json.dumps(cache_key, sort_keys=True)
    
    # Clear cache if force_refresh is True
    if force_refresh:
        _search_papers_impl.cache.delete(cache_key_str)  # type: ignore
    
    # Call the cached implementation
    return _search_papers_impl(
        query=query,
        papers=papers,
        k=k,
        filters=filters,
        sort_by=sort_by,
        start_date=start_date,
        end_date=end_date
    )

def get_paper_embeddings(papers: List[Dict[str, Any]], model_name: str = 'all-mpnet-base-v2') -> np.ndarray:
    """
    Generate embeddings for a list of papers using the specified model.
    
    Args:
        papers: List of paper dictionaries
        model_name: Name of the sentence transformer model to use
        
    Returns:
        Numpy array of embeddings (n_papers, embedding_dim)
    """
    model = get_sentence_transformer(model_name)
    texts = [f"{p.get('title', '')} {p.get('summary', '')}" for p in papers]
    return model.encode(texts, show_progress_bar=True)

def load_arxiv_data(category: str, force_download: bool = False) -> List[Dict[str, Any]]:
    """
    Load arXiv data for the specified category.
    
    Args:
        category: The arXiv category to load data for (e.g., 'cs.CL')
        force_download: If True, force download even if cached data exists
        
    Returns:
        List of paper dictionaries
    """
    global arxiv_dataset
    return arxiv_dataset.load_arxiv_data(category, force_download=force_download)
