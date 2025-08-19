# 📚 Research Paper Assistant

An intelligent research assistant that helps you explore, analyze, and interact with scientific literature using advanced NLP and machine learning techniques. This Streamlit-based application provides a user-friendly interface for academic research and paper discovery.

## ✨ Features

### 🔍 Smart Paper Discovery
- **Semantic Search**: Find relevant research papers using natural language queries
- **Advanced Filtering**: Filter by categories, authors, publication dates, and more
- **AI-Powered Recommendations**: Get personalized paper suggestions based on your interests

### 📖 Enhanced Reading Experience
- **Built-in PDF Viewer**: Read papers directly in the browser
- **Interactive Annotations**: Highlight and take notes on important sections
- **AI Summarization**: Generate concise summaries of research papers
- **Concept Explanation**: Get clear explanations of complex terms and concepts

### 📊 Research Analysis Tools
- **Citation Network Visualization**: Explore connections between papers
- **Trend Analysis**: Track research trends and emerging topics over time
- **Concept Mapping**: Visualize relationships between different research concepts
- **Saved Research**: Organize and manage your research library

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)
- Basic knowledge of Python and command line

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/research-paper-assistant.git
   cd research-paper-assistant
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Update the environment variables in `.env` as needed

### Running the Application

Start the Streamlit app with:
```bash
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`

## 🛠️ Project Structure

```
research-paper-assistant/
├── data/                   # Data storage
│   ├── cache/             # Cached PDFs and processed data
│   ├── cs.CL_papers.json  # Sample arXiv papers dataset
│   └── ...
├── utils/                 # Utility modules
│   ├── auth.py           # Authentication and user management
│   ├── cache_utils.py    # Caching mechanisms
│   ├── data_processing.py # Data processing utilities
│   ├── nlp_utils.py      # NLP processing functions
│   ├── pdf_utils.py      # PDF processing utilities
│   └── trends.py         # Trend analysis functionality
├── .env.example          # Example environment variables
├── .gitignore            # Git ignore file
├── app.py               # Main application file
├── manage_cache.py       # Cache management utilities
├── README.md            # This file
├── requirements.txt      # Python dependencies
└── setup.py             # Project setup file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example` and configure the following:

```
# API Keys (if needed)
OPENAI_API_KEY=your_openai_api_key
ARXIV_API_KEY=your_arxiv_api_key

# Application Settings
DEBUG=True
SECRET_KEY=your-secret-key-here
CACHE_DIR=./data/cache
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ using Streamlit
- Utilizes various open-source libraries (see `requirements.txt`)
- Inspired by the need for better research tools in academia

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact the maintainers.
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory with your configuration:
   ```env
   # API Keys (if needed)
   ARXIV_API_KEY=your_arxiv_api_key
   
   # Application Settings
   MAX_PAPERS=500
   CACHE_DIR=./data/cache
   
   # Model Configurations
   EMBEDDING_MODEL=all-mpnet-base-v2
   SUMMARIZATION_MODEL=facebook/bart-large-cnn
   EXPLANATION_MODEL=google/flan-t5-base
   ```

## 🖥️ Usage

### Starting the Application
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Key Features in Detail

#### 1. Paper Browsing
- Browse papers by category (CS, Physics, Math, etc.)
- View paper details including abstract, authors, and publication date
- Sort by relevance, date, or citation count

#### 2. Smart Search
- Natural language search across paper titles, abstracts, and content
- Filter results by multiple criteria
- Save your search queries for later use

#### 3. Reading Experience
- Built-in PDF viewer with page navigation
- Adjustable text size and contrast
- Night mode for comfortable reading

#### 4. Research Tools
- **Summary Generation**: Get AI-generated summaries of papers
- **Concept Explanation**: Highlight any term to get an explanation
- **Citation Manager**: Track and organize your references
- **Note-taking**: Add personal notes to papers

## 🛠 Advanced Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_PAPERS` | Maximum papers to download per category | 1000 |
| `CACHE_DIR` | Directory for caching papers and embeddings | `./data/cache` |
| `EMBEDDING_MODEL` | Model for text embeddings | `all-mpnet-base-v2` |
| `SUMMARIZATION_MODEL` | Model for paper summarization | `facebook/bart-large-cnn` |
| `EXPLANATION_MODEL` | Model for concept explanations | `google/flan-t5-base` |

### Customizing the Interface
You can customize the look and feel by modifying the `custom.css` file in the `static` directory.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [arXiv](https://arxiv.org/) for open access to research papers
- [Hugging Face](https://huggingface.co/) for pre-trained models
- [Streamlit](https://streamlit.io/) for the web framework
- [Sentence Transformers](https://www.sbert.net/) for text embeddings

## 📧 Contact

For questions or feedback, please open an issue or contact [your-email@example.com](mailto:your-email@example.com)
