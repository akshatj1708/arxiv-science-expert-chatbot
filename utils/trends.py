"""
Research trend analysis and visualization utilities.
"""
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import base64
from io import BytesIO
import streamlit as st

class ResearchTrendAnalyzer:
    """Analyze and visualize research trends from papers."""
    
    def __init__(self, papers: List[Dict]):
        """Initialize with a list of paper dictionaries."""
        self.papers = papers
        self.df = self._prepare_data()
    
    def _prepare_data(self) -> pd.DataFrame:
        """Convert papers to a pandas DataFrame with proper types."""
        data = []
        for paper in self.papers:
            try:
                pub_date = paper.get('published', '')
                if pub_date:
                    if isinstance(pub_date, str):
                        pub_date = pub_date.split('T')[0]  # Extract YYYY-MM-DD
                        year = int(pub_date.split('-')[0])
                    else:
                        year = pub_date.year if hasattr(pub_date, 'year') else 2000
                    
                    data.append({
                        'id': paper.get('id', ''),
                        'title': paper.get('title', ''),
                        'authors': ', '.join(paper.get('authors', [])),
                        'categories': paper.get('categories', []),
                        'published': pub_date,
                        'year': year,
                        'summary': paper.get('summary', ''),
                        'url': paper.get('pdf_url', '')
                    })
            except Exception as e:
                print(f"Error processing paper: {e}")
                continue
        
        return pd.DataFrame(data)
    
    def plot_yearly_trends(self, top_n: int = 10) -> go.Figure:
        """Plot the number of publications per year."""
        if self.df.empty:
            return None
            
        yearly_counts = self.df['year'].value_counts().sort_index()
        
        fig = px.bar(
            x=yearly_counts.index,
            y=yearly_counts.values,
            labels={'x': 'Year', 'y': 'Number of Publications'},
            title='Publications per Year',
            color=yearly_counts.values,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black' if not st.session_state.get('dark_mode', False) else 'white')
        )
        
        return fig
    
    def plot_category_distribution(self) -> go.Figure:
        """Plot the distribution of papers across categories."""
        if self.df.empty:
            return None
            
        # Flatten categories and count occurrences
        categories = [cat for sublist in self.df['categories'] for cat in sublist]
        category_counts = pd.Series(categories).value_counts().head(15)
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title='Top 15 Research Categories',
            hole=0.4
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#000000', width=1))
        )
        
        return fig
    
    def generate_word_cloud(self, text_column: str = 'title', 
                          background_color: str = 'white',
                          width: int = 800, height: int = 400) -> str:
        """Generate a word cloud from text data."""
        if self.df.empty or text_column not in self.df.columns:
            return None
            
        text = ' '.join(self.df[text_column].dropna())
        
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        ).generate(text)
        
        # Convert to base64 for display in Streamlit
        img = wordcloud.to_image()
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f'<img src="data:image/png;base64,{img_str}" width="{width}" height="{height}">'
    
    def topic_modeling(self, num_topics: int = 5, max_features: int = 1000) -> Tuple:
        """Perform topic modeling on paper abstracts."""
        if self.df.empty or 'summary' not in self.df.columns:
            return None, None
        
        # Create TF-IDF features
        tfidf = TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=max_features,
            stop_words='english'
        )
        
        tfidf_vectors = tfidf.fit_transform(self.df['summary'].fillna(''))
        
        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            max_iter=10,
            learning_method='online',
            random_state=42
        )
        
        lda.fit(tfidf_vectors)
        
        # Get top words for each topic
        feature_names = tfidf.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-10 - 1:-1]  # Top 10 words
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx,
                'top_words': ', '.join(top_words),
                'weight': topic.sum()
            })
        
        # Assign dominant topic to each document
        topic_assignments = lda.transform(tfidf_vectors).argmax(axis=1)
        self.df['topic'] = topic_assignments
        
        return pd.DataFrame(topics), lda
    
    def plot_topic_evolution(self, num_topics: int = 5) -> go.Figure:
        """Plot how topics evolve over time."""
        if self.df.empty or 'topic' not in self.df.columns:
            topics_df, _ = self.topic_modeling(num_topics=num_topics)
            if topics_df is None:
                return None
        
        # Group by year and topic
        topic_yearly = pd.crosstab(
            index=self.df['year'],
            columns=self.df['topic'],
            normalize='index'
        )
        
        # Rename columns to show top words
        topic_names = {}
        for topic_id in topic_yearly.columns:
            top_words = self.df[self.df['topic'] == topic_id]['summary']\
                .str.split().explode().value_counts().head(3).index.tolist()
            topic_names[topic_id] = f"Topic {topic_id}: {' '.join(top_words)}"
        
        topic_yearly = topic_yearly.rename(columns=topic_names)
        
        # Plot
        fig = px.area(
            topic_yearly,
            title='Topic Evolution Over Time',
            labels={'value': 'Proportion of Publications', 'year': 'Year'}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black' if not st.session_state.get('dark_mode', False) else 'white'),
            legend_title_text='Topics',
            hovermode='x unified'
        )
        
        return fig
    
    def get_author_network(self, min_collaborations: int = 2) -> Dict:
        """Generate an author collaboration network."""
        if self.df.empty or 'authors' not in self.df.columns:
            return None
        
        import networkx as nx
        
        G = nx.Graph()
        
        # Add nodes and edges for co-authorship
        for _, row in self.df.iterrows():
            authors = row['authors'].split(', ') if isinstance(row['authors'], str) else []
            
            # Add nodes
            for author in authors:
                if author not in G:
                    G.add_node(author, papers=0)
                G.nodes[author]['papers'] += 1
            
            # Add edges for collaborations
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    if G.has_edge(authors[i], authors[j]):
                        G[authors[i]][authors[j]]['weight'] += 1
                    else:
                        G.add_edge(authors[i], authors[j], weight=1)
        
        # Filter nodes by minimum collaborations
        nodes_to_remove = [
            node for node, degree in dict(G.degree()).items() 
            if degree < min_collaborations
        ]
        G.remove_nodes_from(nodes_to_remove)
        
        # Convert to PyVis network
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(G)
        
        # Save to HTML string
        return net.generate_html()


def plot_citation_network(papers: List[Dict]) -> str:
    """Generate a citation network visualization."""
    try:
        from pyvis.network import Network
        import networkx as nx
        
        G = nx.DiGraph()
        
        # Add papers as nodes
        for paper in papers:
            paper_id = paper.get('id', '')
            if paper_id:
                G.add_node(
                    paper_id,
                    label=paper.get('title', '')[:50] + '...',
                    title=f"{paper.get('title', '')}\n"
                          f"Authors: {', '.join(paper.get('authors', []))}\n"
                          f"Year: {paper.get('published', '')[:4]}",
                    group=paper.get('categories', [''])[0],
                    size=10
                )
        
        # Add citation edges (simplified for demo)
        # In a real app, you'd use a citation database or API
        for i, paper1 in enumerate(papers):
            for j, paper2 in enumerate(papers):
                if i != j and np.random.random() < 0.05:  # Random connections for demo
                    G.add_edge(paper1.get('id', ''), paper2.get('id', ''))
        
        # Generate the network
        net = Network(
            height="600px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black"
        )
        
        net.from_nx(G)
        
        # Save to HTML string
        return net.generate_html()
        
    except Exception as e:
        print(f"Error generating citation network: {e}")
        return "<p>Could not generate citation network. Please try again later.</p>"
