import os
import io
import fitz  # PyMuPDF
import base64
import hashlib
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import json
import time
from functools import lru_cache

# Import caching utilities
from .cache_utils import disk_cache, memory_cache

@dataclass
class Annotation:
    """Represents a user annotation on a PDF."""
    id: str
    page: int
    content: str
    x: float  # Normalized coordinates (0-1)
    y: float
    width: float
    height: float
    color: str = "#FFEB3B"  # Default yellow highlight
    created_at: str = ""
    updated_at: str = ""
    tags: List[str] = None

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'page': self.page,
            'content': self.content,
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'color': self.color,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'tags': self.tags or []
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Annotation':
        return cls(
            id=data.get('id', ''),
            page=data.get('page', 0),
            content=data.get('content', ''),
            x=data.get('x', 0),
            y=data.get('y', 0),
            width=data.get('width', 0),
            height=data.get('height', 0),
            color=data.get('color', "#FFEB3B"),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
            tags=data.get('tags', [])
        )

class PDFProcessor:
    """Handle PDF processing including rendering, text extraction, and annotations."""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """Initialize with a cache directory for storing processed PDFs."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.annotation_file = self.cache_dir / "annotations.json"
        self.annotations = self._load_annotations()
        
    def _load_annotations(self) -> Dict[str, List[Dict]]:
        """Load annotations from the annotation file.
        
        Returns:
            Dictionary mapping PDF hashes to lists of annotations
        """
        if not self.annotation_file.exists():
            return {}
            
        try:
            with open(self.annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert the loaded data to Annotation objects
                return {
                    pdf_hash: [Annotation.from_dict(ann) for ann in annotations]
                    for pdf_hash, annotations in data.items()
                }
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading annotations: {e}")
            return {}
    
    def _get_cache_key(self, pdf_url: str) -> str:
        """Generate a cache key from PDF URL."""
        return hashlib.md5(pdf_url.encode('utf-8')).hexdigest()
    
    @disk_cache(ttl=604800)  # Cache for 7 days
    def _download_pdf(self, pdf_url: str) -> bytes:
        """Download PDF content with caching."""
        response = requests.get(pdf_url)
        response.raise_for_status()
        return response.content
    
    def get_pdf_document(self, pdf_url: str):
        """Load a PDF document from a URL with caching."""
        try:
            # Get PDF content (from cache or download)
            pdf_content = self._download_pdf(pdf_url)
            
            # Save to disk cache
            cache_key = self._get_cache_key(pdf_url)
            cache_path = self.cache_dir / f"{cache_key}.pdf"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_path, 'wb') as f:
                f.write(pdf_content)
                
            # Return as a fitz document
            return fitz.open(stream=pdf_content, filetype='pdf')
            
        except Exception as e:
            # Fallback to disk cache if download fails
            if cache_path.exists():
                return fitz.open(cache_path)
            raise
    
    @disk_cache(ttl=86400)  # Cache for 1 day
    def _render_page_impl(self, pdf_url: str, page_num: int, dpi: int) -> bytes:
        """Internal implementation of page rendering with caching."""
        doc = self.get_pdf_document(pdf_url)
        page = doc.load_page(page_num)
        
        # Calculate zoom factor based on DPI
        zoom = dpi / 72  # 72 is the default DPI
        mat = fitz.Matrix(zoom, zoom)
        
        # Render page to an image
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes('png')
    
    def render_page_as_image(
            self, 
            pdf_url: str, 
            page_num: int = 0, 
            dpi: int = 150,
            width: int = 800
        ) -> str:
        """Render a PDF page as an image with caching."""
        try:
            # Get the rendered image from cache or generate it
            img_data = self._render_page_impl(pdf_url, page_num, dpi)
            
            # Convert to base64 for HTML display
            return f"data:image/png;base64,{base64.b64encode(img_data).decode()}"
            
        except Exception as e:
            print(f"Error rendering PDF page: {e}")
            # Return a simple error message as base64 encoded image
            try:
                from PIL import Image, ImageDraw, ImageFont
                img = Image.new('RGB', (width, int(width * 1.414)), color=(240, 240, 240))
                d = ImageDraw.Draw(img)
                try:
                    font = ImageFont.load_default()
                    d.text((10, 10), f"Error loading PDF: {str(e)[:100]}", fill=(255, 0, 0), font=font)
                except:
                    d.text((10, 10), "Error loading PDF", fill=(255, 0, 0))
                
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                return f"data:image/png;base64,{base64.b64encode(img_byte_arr.getvalue()).decode()}"
            except Exception as img_err:
                # If even the error image generation fails, return a simple error string
                return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==", 1.414
    
    @disk_cache(ttl=86400)  # Cache for 1 day
    def _extract_text_impl(
            self, 
            pdf_url: str, 
            page_num: Optional[int],
            rect: Optional[Tuple[float, float, float, float]]
        ) -> str:
        """Internal implementation of text extraction with caching."""
        doc = self.get_pdf_document(pdf_url)
        
        if page_num is not None:
            page = doc.load_page(page_num)
            if rect:
                # Extract text from specific rectangle
                return page.get_text("text", clip=rect)
            return page.get_text("text")
        else:
            # Extract all text from the document
            return "\n\n".join(page.get_text("text") for page in doc)
    
    def extract_text(
            self, 
            pdf_url: str, 
            page_num: Optional[int] = None,
            rect: Optional[Tuple[float, float, float, float]] = None
        ) -> str:
        """Extract text from a PDF page or a specific region with caching."""
        try:
            text = self._extract_text_impl(pdf_url, page_num, rect)
            return text.strip() if text else ""
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def search_in_pdf(
        self, 
        pdf_url: str, 
        query: str, 
        case_sensitive: bool = False
    ) -> List[Dict]:
        """Search for text in a PDF and return matching locations."""
        try:
            doc = self.get_pdf_document(pdf_url)
            results = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_instances = page.search_for(
                    query,
                    quads=False,
                    flags=0 if case_sensitive else fitz.TEXT_PRESERVE_WHITESPACE
                )
                
                for inst in text_instances:
                    # Convert to normalized coordinates (0-1)
                    page_rect = page.rect
                    x0 = (inst.x0 - page_rect.x0) / page_rect.width
                    y0 = (inst.y0 - page_rect.y0) / page_rect.height
                    x1 = (inst.x1 - page_rect.x0) / page_rect.width
                    y1 = (inst.y1 - page_rect.y0) / page_rect.height
                    
                    # Extract the matching text
                    match_text = page.get_text("text", clip=inst).strip()
                    
                    results.append({
                        'page': page_num,
                        'x': x0,
                        'y': y0,
                        'width': x1 - x0,
                        'height': y1 - y0,
                        'text': match_text,
                        'snippet': self._get_surrounding_text(page, inst)
                    })
            
            return results
            
        except Exception as e:
            print(f"Error searching PDF: {e}")
            return []
    
    def _get_surrounding_text(self, page, rect, chars_before: int = 50, chars_after: int = 50) -> str:
        """Get text surrounding a rectangle on a page."""
        try:
            # Get a slightly larger rectangle
            margin = 100  # points
            expanded_rect = fitz.Rect(
                max(0, rect.x0 - margin),
                max(0, rect.y0 - margin),
                min(page.rect.width, rect.x1 + margin),
                min(page.rect.height, rect.y1 + margin)
            )
            
            text = page.get_text("text", clip=expanded_rect)
            if not text:
                return ""
                
            # Find the position of our original text
            match_text = page.get_text("text", clip=rect).strip()
            if not match_text:
                return text
                
            pos = text.find(match_text)
            if pos == -1:
                return text
                
            # Extract surrounding text
            start = max(0, pos - chars_before)
            end = min(len(text), pos + len(match_text) + chars_after)
            
            # Add ellipsis if we're not at the start/end
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(text) else ""
            
            return f"{prefix}{text[start:end].strip()}{suffix}"
            
        except Exception:
            return ""
    
    def create_annotation(
        self, 
        pdf_url: str, 
        annotation_data: Dict
    ) -> Optional[Annotation]:
        """Create a new annotation on a PDF page."""
        try:
            # Generate a unique ID if not provided
            if 'id' not in annotation_data or not annotation_data['id']:
                import uuid
                annotation_data['id'] = str(uuid.uuid4())
            
            # Set timestamps
            from datetime import datetime
            now = datetime.utcnow().isoformat()
            if 'created_at' not in annotation_data or not annotation_data['created_at']:
                annotation_data['created_at'] = now
            annotation_data['updated_at'] = now
            
            # Create annotation object
            annotation = Annotation.from_dict(annotation_data)
            
            # In a real application, you would save this to a database
            # For now, we'll just return the annotation
            return annotation
            
        except Exception as e:
            print(f"Error creating annotation: {e}")
            return None
    
    def get_annotations(
        self, 
        pdf_url: str, 
        page_num: Optional[int] = None
    ) -> List[Annotation]:
        """Get all annotations for a PDF (or a specific page)."""
        # In a real application, this would query a database
        # For now, return an empty list
        return []
    
    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete an annotation by ID."""
        # In a real application, this would delete from a database
        # For now, just return True to indicate success
        return True
