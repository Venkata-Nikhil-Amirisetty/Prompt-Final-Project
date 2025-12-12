"""
Documentation scraper for Python official documentation.

This module handles scraping, cleaning, and storing Python documentation
from the official Python tutorial website.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base URLs for Python documentation
TUTORIAL_BASE_URL = "https://docs.python.org/3/tutorial/"
LIBRARY_BASE_URL = "https://docs.python.org/3/library/"
REFERENCE_BASE_URL = "https://docs.python.org/3/reference/"

# Rate limiting: 1 second between requests
REQUEST_DELAY = 1.0


def scrape_library_reference(
    output_dir: str = "data",
    delay: float = REQUEST_DELAY
) -> List[Dict]:
    """
    Scrape Python Standard Library Reference documentation.
    
    This includes popular and commonly used modules that provide
    the most value for a documentation assistant.
    
    Args:
        output_dir: Directory to save scraped content
        delay: Delay between requests in seconds
        
    Returns:
        List of dictionaries containing scraped content
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Popular standard library modules - most commonly used
    library_urls = [
        # File and Directory Operations
        "https://docs.python.org/3/library/os.html",
        "https://docs.python.org/3/library/pathlib.html",
        "https://docs.python.org/3/library/shutil.html",
        "https://docs.python.org/3/library/glob.html",
        
        # System and Environment
        "https://docs.python.org/3/library/sys.html",
        "https://docs.python.org/3/library/platform.html",
        
        # Data Serialization
        "https://docs.python.org/3/library/json.html",
        "https://docs.python.org/3/library/pickle.html",
        
        # Date and Time
        "https://docs.python.org/3/library/datetime.html",
        "https://docs.python.org/3/library/time.html",
        
        # Data Structures
        "https://docs.python.org/3/library/collections.html",
        "https://docs.python.org/3/library/heapq.html",
        "https://docs.python.org/3/library/bisect.html",
        "https://docs.python.org/3/library/array.html",
        
        # String and Text Processing
        "https://docs.python.org/3/library/string.html",
        "https://docs.python.org/3/library/re.html",  # Regular expressions
        "https://docs.python.org/3/library/textwrap.html",
        
        # Iteration and Functional Programming
        "https://docs.python.org/3/library/itertools.html",
        "https://docs.python.org/3/library/functools.html",
        
        # Math and Statistics
        "https://docs.python.org/3/library/math.html",
        "https://docs.python.org/3/library/statistics.html",
        "https://docs.python.org/3/library/random.html",
        
        # File I/O
        "https://docs.python.org/3/library/io.html",
        "https://docs.python.org/3/library/csv.html",
        
        # URL and Web
        "https://docs.python.org/3/library/urllib.html",
        "https://docs.python.org/3/library/urllib.parse.html",
        
        # Utilities
        "https://docs.python.org/3/library/argparse.html",
        "https://docs.python.org/3/library/logging.html",
        "https://docs.python.org/3/library/hashlib.html",
        "https://docs.python.org/3/library/base64.html",
        
        # Advanced
        "https://docs.python.org/3/library/typing.html",  # Type hints
        "https://docs.python.org/3/library/contextlib.html",  # Context managers
        "https://docs.python.org/3/library/abc.html",  # Abstract base classes
    ]
    
    logger.info(f"Scraping {len(library_urls)} standard library modules...")
    return scrape_custom_urls(library_urls, output_dir, delay, prefix="lib")


def scrape_language_reference(
    output_dir: str = "data",
    delay: float = REQUEST_DELAY
) -> List[Dict]:
    """
    Scrape Python Language Reference documentation.
    
    Covers advanced language features like decorators, generators,
    metaclasses, and the complete language specification.
    
    Args:
        output_dir: Directory to save scraped content
        delay: Delay between requests in seconds
        
    Returns:
        List of dictionaries containing scraped content
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Language reference sections
    reference_urls = [
        # Data Model (includes decorators, descriptors, etc.)
        "https://docs.python.org/3/reference/datamodel.html",
        
        # Execution Model
        "https://docs.python.org/3/reference/executionmodel.html",
        
        # Expressions
        "https://docs.python.org/3/reference/expressions.html",
        
        # Simple Statements
        "https://docs.python.org/3/reference/simple_stmts.html",
        
        # Compound Statements (functions, classes, decorators)
        "https://docs.python.org/3/reference/compound_stmts.html",
        
        # Top-level Components
        "https://docs.python.org/3/reference/toplevel_components.html",
        
        # Glossary (definitions)
        "https://docs.python.org/3/glossary.html",
    ]
    
    logger.info(f"Scraping {len(reference_urls)} language reference sections...")
    return scrape_custom_urls(reference_urls, output_dir, delay, prefix="ref")


def scrape_advanced_topics(
    output_dir: str = "data",
    delay: float = REQUEST_DELAY
) -> List[Dict]:
    """
    Scrape advanced Python topics from various documentation sections.
    
    Args:
        output_dir: Directory to save scraped content
        delay: Delay between requests in seconds
        
    Returns:
        List of dictionaries containing scraped content
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Advanced topics from different sections
    advanced_urls = [
        # Generators and Iterators (from tutorial)
        "https://docs.python.org/3/tutorial/classes.html#generators",
        "https://docs.python.org/3/tutorial/classes.html#generator-expressions",
        "https://docs.python.org/3/tutorial/classes.html#iterators",
        
        # Async/Await
        "https://docs.python.org/3/library/asyncio.html",
    ]
    
    return scrape_custom_urls(advanced_urls, output_dir, delay, prefix="adv")


def scrape_custom_urls(
    urls: List[str],
    output_dir: str = "data",
    delay: float = REQUEST_DELAY,
    prefix: str = "doc"
) -> List[Dict]:
    """
    Scrape custom list of URLs.
    
    Args:
        urls: List of URLs to scrape
        output_dir: Directory to save scraped content
        delay: Delay between requests in seconds
        
    Returns:
        List of dictionaries containing scraped content
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    scraped_data = []
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for i, url in enumerate(urls, 1):
        try:
            logger.info(f"Scraping {prefix} {i}/{len(urls)}: {url}")
            
            response = session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove navigation, footer, etc.
            for element in soup.find_all(['nav', 'footer', 'header']):
                element.decompose()
            for element in soup.find_all(['script', 'style']):
                element.decompose()
            
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Python Documentation"
            
            main_content = soup.find('div', class_='body') or soup.find('main') or soup.find('body')
            
            if main_content:
                content = main_content.get_text(separator='\n', strip=True)
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                content = '\n'.join(lines)
                
                doc_data = {
                    'url': url,
                    'title': title_text,
                    'content': content,
                    'date_scraped': datetime.now().isoformat(),
                    'content_length': len(content)
                }
                
                scraped_data.append(doc_data)
                
                filename = f"{prefix}_{i:03d}_{urlparse(url).path.split('/')[-1] or 'index'}.json"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(doc_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved: {filepath} ({len(content)} characters)")
            
            if i < len(urls):
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            continue
    
    return scraped_data


def scrape_python_docs(
    base_url: str = TUTORIAL_BASE_URL,
    max_pages: int = 100,
    output_dir: str = "data",
    delay: float = REQUEST_DELAY,
    include_advanced: bool = True
) -> List[Dict]:
    """
    Scrape Python documentation pages from the tutorial section.
    
    Args:
        base_url: Base URL for Python documentation tutorial
        max_pages: Maximum number of pages to scrape
        output_dir: Directory to save scraped content
        delay: Delay between requests in seconds
        
    Returns:
        List of dictionaries containing scraped content with metadata
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Pages to scrape - expanded tutorial sections including advanced topics
    pages_to_scrape = [
        # Basic Tutorial
        "index.html",
        "introduction.html",
        "interpreter.html",
        "introduction.html#informal-introduction",
        
        # Data Structures
        "datastructures.html",
        "datastructures.html#more-on-lists",
        "datastructures.html#using-lists-as-stacks",
        "datastructures.html#using-lists-as-queues",
        "datastructures.html#list-comprehensions",
        "datastructures.html#nested-list-comprehensions",
        "datastructures.html#the-del-statement",
        "datastructures.html#tuples-and-sequences",
        "datastructures.html#sets",
        "datastructures.html#dictionaries",
        "datastructures.html#looping-techniques",
        "datastructures.html#more-on-conditions",
        "datastructures.html#comparing-sequences-and-other-types",
        
        # Control Flow
        "controlflow.html",
        "controlflow.html#if-statements",
        "controlflow.html#for-statements",
        "controlflow.html#the-range-function",
        "controlflow.html#break-and-continue-statements",
        "controlflow.html#pass-statements",
        "controlflow.html#match-statements",
        "controlflow.html#defining-functions",
        
        # Functions (Advanced)
        "functions.html",
        "functions.html#more-on-defining-functions",
        "functions.html#default-argument-values",
        "functions.html#keyword-arguments",
        "functions.html#special-parameters",
        "functions.html#arbitrary-argument-lists",
        "functions.html#unpacking-argument-lists",
        "functions.html#lambda-expressions",
        "functions.html#documentation-strings",
        "functions.html#function-annotations",
        
        # Data Structures (Advanced)
        "datastructures.html#list-comprehensions",
        "datastructures.html#nested-list-comprehensions",
        
        # Modules
        "modules.html",
        "modules.html#more-on-modules",
        "modules.html#standard-modules",
        "modules.html#the-dir-function",
        "modules.html#packages",
        "modules.html#intra-package-references",
        "modules.html#packages-in-multiple-directories",
        
        # Input/Output
        "inputoutput.html",
        "inputoutput.html#fancier-output-formatting",
        "inputoutput.html#old-string-formatting",
        "inputoutput.html#reading-and-writing-files",
        "inputoutput.html#methods-of-file-objects",
        "inputoutput.html#saving-structured-data-with-json",
        
        # Errors and Exceptions
        "errors.html",
        "errors.html#syntax-errors",
        "errors.html#exceptions",
        "errors.html#handling-exceptions",
        "errors.html#raising-exceptions",
        "errors.html#exception-chaining",
        "errors.html#user-defined-exceptions",
        "errors.html#defining-clean-up-actions",
        
        # Classes
        "classes.html",
        "classes.html#a-word-about-names-and-objects",
        "classes.html#python-scopes-and-namespaces",
        "classes.html#a-first-look-at-classes",
        "classes.html#class-objects",
        "classes.html#instance-objects",
        "classes.html#method-objects",
        "classes.html#class-and-instance-variables",
        "classes.html#random-remarks",
        "classes.html#inheritance",
        "classes.html#multiple-inheritance",
        "classes.html#private-variables",
        "classes.html#odds-and-ends",
        "classes.html#iterators",
        "classes.html#generators",
        "classes.html#generator-expressions",
        
        # Standard Library
        "stdlib.html",
        "stdlib.html#os-interface",
        "stdlib.html#file-wildcards",
        "stdlib.html#command-line-arguments",
        "stdlib.html#error-output-redirection-and-program-termination",
        "stdlib.html#string-pattern-matching",
        "stdlib.html#mathematics",
        "stdlib.html#internet-access",
        "stdlib.html#dates-and-times",
        "stdlib.html#data-compression",
        "stdlib.html#performance-measurement",
        "stdlib.html#quality-control",
        "stdlib.html#batteries-included",
        
        "stdlib2.html",
        "stdlib2.html#output-formatting",
        "stdlib2.html#templating",
        "stdlib2.html#working-with-binary-data-record-layouts",
        "stdlib2.html#multi-threading",
        "stdlib2.html#logging",
        "stdlib2.html#weak-references",
        "stdlib2.html#tools-for-working-with-lists",
        "stdlib2.html#decimal-floating-point-arithmetic",
        
        # Advanced Topics (from Language Reference)
        # Note: These are from docs.python.org/3/reference/ but we'll add tutorial equivalents
    ]
    
    # Note: Advanced topics like decorators are handled separately via scrape_advanced_topics()
    
    scraped_data = []
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for i, page in enumerate(pages_to_scrape[:max_pages], 1):
        try:
            # Construct full URL
            if page.startswith('http'):
                url = page
            else:
                url = urljoin(base_url, page)
            
            logger.info(f"Scraping page {i}/{max_pages}: {url}")
            
            # Fetch page
            response = session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove navigation, footer, and other non-content elements
            for element in soup.find_all(['nav', 'footer', 'header']):
                element.decompose()
            
            # Remove script and style tags
            for element in soup.find_all(['script', 'style']):
                element.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Python Documentation"
            
            # Extract main content
            main_content = soup.find('div', class_='body') or soup.find('main') or soup.find('body')
            
            if main_content:
                # Get text content
                content = main_content.get_text(separator='\n', strip=True)
                
                # Clean up content
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                content = '\n'.join(lines)
                
                # Create metadata
                doc_data = {
                    'url': url,
                    'title': title_text,
                    'content': content,
                    'date_scraped': datetime.now().isoformat(),
                    'content_length': len(content)
                }
                
                scraped_data.append(doc_data)
                
                # Save individual file
                filename = f"doc_{i:03d}_{urlparse(url).path.split('/')[-1] or 'index'}.json"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(doc_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved: {filepath} ({len(content)} characters)")
            else:
                logger.warning(f"No content found for {url}")
            
            # Rate limiting
            if i < len(pages_to_scrape[:max_pages]):
                time.sleep(delay)
                
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            continue
    
    # Scrape library reference if requested
    if include_advanced:
        logger.info("Scraping standard library reference...")
        try:
            library_data = scrape_library_reference(output_dir, delay)
            scraped_data.extend(library_data)
            logger.info(f"Added {len(library_data)} library reference pages")
        except Exception as e:
            logger.warning(f"Error scraping library reference: {e}")
        
        logger.info("Scraping language reference...")
        try:
            reference_data = scrape_language_reference(output_dir, delay)
            scraped_data.extend(reference_data)
            logger.info(f"Added {len(reference_data)} language reference pages")
        except Exception as e:
            logger.warning(f"Error scraping language reference: {e}")
        
        logger.info("Scraping additional advanced topics...")
        try:
            advanced_data = scrape_advanced_topics(output_dir, delay)
            scraped_data.extend(advanced_data)
            logger.info(f"Added {len(advanced_data)} advanced topic pages")
        except Exception as e:
            logger.warning(f"Error scraping advanced topics: {e}")
    
    # Save combined data
    combined_filepath = os.path.join(output_dir, "all_docs.json")
    with open(combined_filepath, 'w', encoding='utf-8') as f:
        json.dump(scraped_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Scraping complete. Total pages: {len(scraped_data)}")
    return scraped_data


def load_scraped_data(data_dir: str = "data") -> List[Dict]:
    """
    Load previously scraped documentation data.
    
    Args:
        data_dir: Directory containing scraped JSON files
        
    Returns:
        List of dictionaries containing scraped content
    """
    combined_filepath = os.path.join(data_dir, "all_docs.json")
    
    if os.path.exists(combined_filepath):
        with open(combined_filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # If combined file doesn't exist, load individual files
    scraped_data = []
    data_path = Path(data_dir)
    
    if data_path.exists():
        for json_file in data_path.glob("doc_*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    scraped_data.append(json.load(f))
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
    
    return scraped_data


if __name__ == "__main__":
    # Test scraper
    data = scrape_python_docs(max_pages=20)
    print(f"Scraped {len(data)} pages")

