"""
Tests for the scraper module.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from src.scraper import load_scraped_data, scrape_python_docs


class TestScraper:
    """Test cases for scraper functionality."""
    
    def test_url_fetching(self):
        """Test that URLs can be fetched."""
        # Mock requests to avoid actual network calls in tests
        with patch('src.scraper.requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.content = b'<html><body><div class="body">Test content</div></body></html>'
            mock_response.raise_for_status = Mock()
            
            mock_session_instance = Mock()
            mock_session_instance.get.return_value = mock_response
            mock_session_instance.headers = {}
            mock_session.return_value = mock_session_instance
            
            with tempfile.TemporaryDirectory() as tmpdir:
                docs = scrape_python_docs(
                    base_url="https://docs.python.org/3/tutorial/",
                    max_pages=1,
                    output_dir=tmpdir
                )
                
                assert len(docs) > 0
                assert 'url' in docs[0]
                assert 'content' in docs[0]
    
    def test_content_extraction(self):
        """Test that content is properly extracted."""
        html_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <div class="body">
                    <h1>Test Title</h1>
                    <p>Test paragraph content.</p>
                </div>
            </body>
        </html>
        """
        
        with patch('src.scraper.requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.content = html_content.encode()
            mock_response.raise_for_status = Mock()
            
            mock_session_instance = Mock()
            mock_session_instance.get.return_value = mock_response
            mock_session_instance.headers = {}
            mock_session.return_value = mock_session_instance
            
            with tempfile.TemporaryDirectory() as tmpdir:
                docs = scrape_python_docs(
                    base_url="https://test.com/",
                    max_pages=1,
                    output_dir=tmpdir
                )
                
                if docs:
                    content = docs[0].get('content', '')
                    assert 'Test paragraph content' in content
                    assert len(content) > 0
    
    def test_metadata_saving(self):
        """Test that metadata is properly saved."""
        with patch('src.scraper.requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.content = b'<html><body><div class="body">Content</div></body></html>'
            mock_response.raise_for_status = Mock()
            
            mock_session_instance = Mock()
            mock_session_instance.get.return_value = mock_response
            mock_session_instance.headers = {}
            mock_session.return_value = mock_session_instance
            
            with tempfile.TemporaryDirectory() as tmpdir:
                docs = scrape_python_docs(
                    base_url="https://test.com/",
                    max_pages=1,
                    output_dir=tmpdir
                )
                
                if docs:
                    doc = docs[0]
                    assert 'url' in doc
                    assert 'title' in doc
                    assert 'date_scraped' in doc
                    assert 'content_length' in doc
    
    def test_rate_limiting(self):
        """Test that rate limiting works."""
        import time
        
        with patch('src.scraper.requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.content = b'<html><body><div class="body">Content</div></body></html>'
            mock_response.raise_for_status = Mock()
            
            mock_session_instance = Mock()
            mock_session_instance.get.return_value = mock_response
            mock_session_instance.headers = {}
            mock_session.return_value = mock_session_instance
            
            with patch('src.scraper.time.sleep') as mock_sleep:
                with tempfile.TemporaryDirectory() as tmpdir:
                    scrape_python_docs(
                        base_url="https://test.com/",
                        max_pages=3,
                        output_dir=tmpdir,
                        delay=1.0
                    )
                    
                    # Should sleep between requests
                    assert mock_sleep.call_count >= 2
    
    def test_error_handling(self):
        """Test error handling for bad URLs."""
        with patch('src.scraper.requests.Session') as mock_session:
            mock_session_instance = Mock()
            mock_session_instance.get.side_effect = requests.RequestException("Connection error")
            mock_session_instance.headers = {}
            mock_session.return_value = mock_session_instance
            
            with tempfile.TemporaryDirectory() as tmpdir:
                docs = scrape_python_docs(
                    base_url="https://invalid-url-12345.com/",
                    max_pages=1,
                    output_dir=tmpdir
                )
                
                # Should handle error gracefully
                assert isinstance(docs, list)
    
    def test_load_scraped_data(self):
        """Test loading previously scraped data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data file
            test_data = [
                {
                    'url': 'https://test.com/page1',
                    'title': 'Test Page 1',
                    'content': 'Test content 1',
                    'date_scraped': '2024-01-01',
                    'content_length': 14
                }
            ]
            
            all_docs_file = Path(tmpdir) / "all_docs.json"
            with open(all_docs_file, 'w') as f:
                json.dump(test_data, f)
            
            # Load data
            loaded = load_scraped_data(data_dir=tmpdir)
            assert len(loaded) == 1
            assert loaded[0]['title'] == 'Test Page 1'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

