import streamlit as st
import pysnow
import re
import html
from typing import List, Dict, Optional
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import requests
import logging
from datetime import datetime, timedelta

st.set_page_config(
    page_title="SharePoint KB Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embedding function
@st.cache_resource
def get_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Custom synonym mapping
SYNONYM_MAP = {
    r'\bie\b': 'internet explorer',
    r'\bwin\b': 'windows',
    r'\bwin8\b': 'windows 8',
    r'\bwin10\b': 'windows 10',
    r'\bver\b': 'version',
    r'\bapp\b': 'application',
    r'\bconfig\b': 'configuration'
}

def format_instance_url(url: str) -> str:
    """Format ServiceNow instance URL to ensure correct format."""
    if not url:
        raise ValueError("Instance URL cannot be empty")
    
    url = url.lower().strip()
    url = url.replace('http://', '').replace('https://', '')
    url = url.replace('.service-now.com', '')
    return url.rstrip('/')

def expand_synonyms(text: str) -> str:
    """Replace abbreviations with full terms using regex."""
    if not text:
        return ""
    
    for pattern, replacement in SYNONYM_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def preprocess_text(text: str) -> str:
    """Preprocess text for better search results."""
    if not text:
        return ""
    
    # Clean text
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s.-]', ' ', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = expand_synonyms(text.lower())
    return text.strip()

def clean_html_content(content: str) -> str:
    """Clean and format HTML content for better display."""
    if not content:
        return ""
    
    # Convert common HTML elements to Markdown
    content = html.unescape(content)
    content = re.sub(r'</?p>', '\n\n', content)
    content = re.sub(r'</?strong>', '**', content)
    content = re.sub(r'</?em>', '*', content)
    content = re.sub(r'<li>', '\n• ', content)
    content = re.sub(r'</li>', '', content)
    content = re.sub(r'</?[ou]l>', '\n', content)
    content = re.sub(r'<br\s*/?>', '\n', content)
    content = re.sub(r'<[^>]+>', '', content)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content.strip()

def initialize_snow_client(instance_url: str, username: str, password: str) -> Optional[pysnow.Client]:
    """Initialize ServiceNow client with validation."""
    try:
        if not all([instance_url, username, password]):
            raise ValueError("All ServiceNow credentials are required")
            
        instance = format_instance_url(instance_url)
        if not re.match(r'^[a-zA-Z0-9-]+$', instance):
            raise ValueError("Invalid instance name format")
            
        client = pysnow.Client(
            instance=instance,
            user=username,
            password=password
        )
        
        # Test connection
        test_resource = client.resource(api_path='/table/sys_user')
        test_resource.get(query={'sysparm_limit': 1})
        logger.info("Successfully connected to ServiceNow")
        return client
        
    except Exception as e:
        logger.error(f"ServiceNow connection failed: {str(e)}")
        st.error(f"ServiceNow connection failed: {str(e)}")
        return None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_graph_token(tenant_id: str, client_id: str, client_secret: str) -> Optional[str]:
    """Get access token for Microsoft Graph API."""
    try:
        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        
        token_data = {
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret,
            'scope': 'https://graph.microsoft.com/.default'
        }
        
        response = requests.post(token_url, data=token_data)
        response.raise_for_status()
        
        return response.json().get('access_token')
        
    except Exception as e:
        logger.error(f"Failed to get access token: {str(e)}")
        return None

def initialize_sharepoint_client(tenant_id: str, client_id: str, client_secret: str) -> Optional[Dict]:
    """Initialize Microsoft Graph API access."""
    try:
        if not all([tenant_id, client_id, client_secret]):
            raise ValueError("All SharePoint credentials are required")
            
        access_token = get_graph_token(tenant_id, client_id, client_secret)
        if not access_token:
            raise ValueError("Failed to get access token")
            
        return {
            'token': access_token,
            'headers': {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
        }
        
    except Exception as e:
        logger.error(f"SharePoint connection failed: {str(e)}")
        st.error(f"SharePoint connection failed: {str(e)}")
        return None

def get_sharepoint_documents(client: Dict, site_url: str, library_name: str) -> List[Dict]:
    """Fetch documents from SharePoint library using Graph API."""
    try:
        if not all([client, site_url, library_name]):
            raise ValueError("Client, site URL and library name are required")
        
        # Extract site name from URL
        site_parts = site_url.strip('/').split('/')
        site_name = site_parts[-1] if site_parts[-1] else site_parts[-2]
        
        headers = client['headers']
        base_url = "https://graph.microsoft.com/v1.0"
        
        # Get site ID
        site_response = requests.get(
            f"{base_url}/sites/{site_name}",
            headers=headers
        )
        site_response.raise_for_status()
        site_id = site_response.json()['id']
        
        # Get drives (document libraries)
        drives_response = requests.get(
            f"{base_url}/sites/{site_id}/drives",
            headers=headers
        )
        drives_response.raise_for_status()
        drives = drives_response.json()['value']
        
        # Find the specified library
        drive_id = None
        for drive in drives:
            if drive['name'].lower() == library_name.lower():
                drive_id = drive['id']
                break
                
        if not drive_id:
            raise ValueError(f"Document library '{library_name}' not found")
        
        # Get files from the drive
        files_response = requests.get(
            f"{base_url}/drives/{drive_id}/root/children",
            headers=headers
        )
        files_response.raise_for_status()
        files = files_response.json()['value']
        
        documents = []
        for file in files:
            if '@microsoft.graph.downloadUrl' in file:
                try:
                    # Download file content
                    content_response = requests.get(file['@microsoft.graph.downloadUrl'])
                    if content_response.status_code == 200:
                        content = content_response.content.decode('utf-8')
                        doc = {
                            'title': file.get('name', ''),
                            'filename': file.get('name', ''),
                            'content': content,
                            'url': file.get('webUrl', '')
                        }
                        documents.append(doc)
                except UnicodeDecodeError:
                    logger.warning(f"Skipping binary file: {file.get('name')}")
                    continue
                    
        logger.info(f"Retrieved {len(documents)} documents from SharePoint")
        return documents
        
    except Exception as e:
        logger.error(f"Error fetching SharePoint documents: {str(e)}")
        st.error(f"Error fetching SharePoint documents: {str(e)}")
        return []

def create_kb_documents(sharepoint_docs: List[Dict]) -> List[Document]:
    """Convert SharePoint documents to LangChain documents."""
    try:
        if not sharepoint_docs:
            raise ValueError("No documents provided for processing")
        
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        for doc in sharepoint_docs:
            processed_text = preprocess_text(doc['content'])
            
            metadata = {
                'title': doc['title'],
                'filename': doc['filename'],
                'url': doc['url'],
                'original_text': doc['content']
            }
            
            chunks = text_splitter.create_documents(
                texts=[processed_text],
                metadatas=[metadata]
            )
            documents.extend(chunks)
        
        logger.info(f"Created {len(documents)} document chunks")
        return documents
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        st.error(f"Error processing documents: {str(e)}")
        return []

def initialize_vector_store(documents: List[Document]) -> Optional[Chroma]:
    """Initialize Chroma vector store with documents."""
    try:
        if not documents:
            raise ValueError("No documents provided for vector store")
            
        embeddings = get_embeddings()
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./kb_store"
        )
        logger.info("Successfully initialized vector store")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        st.error(f"Error initializing vector store: {str(e)}")
        return None

def search_articles(vector_store: Chroma, query: str, n_results: int = 5) -> List[Dict]:
    """Search articles using similarity search."""
    try:
        if not vector_store or not query:
            raise ValueError("Vector store and query are required")
        
        processed_query = preprocess_text(query)
        
        docs = vector_store.similarity_search_with_relevance_scores(
            query=processed_query,
            k=n_results
        )
        
        results = []
        seen_urls = set()
        
        for doc, score in docs:
            if doc.metadata['url'] in seen_urls:
                continue
                
            seen_urls.add(doc.metadata['url'])
            
            result = {
                'title': doc.metadata['title'],
                'filename': doc.metadata['filename'],
                'url': doc.metadata['url'],
                'content': doc.metadata['original_text'],
                'similarity': score
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} relevant articles")
        return results
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        st.error(f"Search error: {str(e)}")
        return []

def get_incident_details(client: pysnow.Client, incident_number: str) -> Optional[Dict]:
    """Fetch incident details from ServiceNow."""
    try:
        if not client or not incident_number:
            raise ValueError("Client and incident number are required")
            
        incident_table = client.resource(api_path='/table/incident')
        response = incident_table.get(query={'number': incident_number})
        incident = response.one()
        logger.info(f"Successfully retrieved incident {incident_number}")
        return incident
        
    except Exception as e:
        logger.error(f"Error fetching incident: {str(e)}")
        st.error(f"Error fetching incident: {str(e)}")
        return None

def create_search_query_from_incident(incident: Dict) -> str:
    """Create search query text from incident details."""
    if not incident:
        return ""
        
    fields = [
        incident.get('short_description', ''),
        incident.get('description', ''),
        incident.get('comments', ''),
        incident.get('work_notes', '')
    ]
    return ' '.join(filter(None, fields))

def post_work_note(client: pysnow.Client, incident_sys_id: str, article_details: Dict) -> bool:
    """Post a work note to an incident with KB article content."""
    try:
        if not all([client, incident_sys_id, article_details]):
            raise ValueError("Client, incident ID and article details are required")
            
        incident_table = client.resource(api_path='/table/incident')
        
        note_content = f"""
SharePoint KB Article Reference:
Title: {article_details['title']}
Filename: {article_details['filename']}
URL: {article_details['url']}
Relevance Score: {article_details['similarity']:.2f}

Article Content:
{clean_html_content(article_details['content'])}

(Added by KB Assistant)
"""
        
        response = incident_table.update(
            query={'sys_id': incident_sys_id},
            payload={"work_notes": note_content}
        )
        
        logger.info(f"Successfully posted work note to incident {incident_sys_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error posting work note: {str(e)}")
        st.error(f"Error posting work note: {str(e)}")
        return False

def display_incident_details(incident: Dict):
    """Display incident details in the UI."""
    if not incident:
        return
        
    st.subheader("Current Incident Details")
    st.write(f"Number: {incident.get('number')}")
    st.write(f"Short Description: {incident.get('short_description')}")
    
    with st.expander("Full Description"):
        st.write(incident.get('description', 'No description available'))

def display_results(results: List[Dict]):
    """Display search results in the UI."""
    if not results:
        return
        
    for result in results:
        title = result['title'] or result['filename']
        score = f"Relevance Score: {result['similarity']:.2f}"
        
        with st.expander(f"{title} ({score})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Article Content")
                content = clean_html_content(result['content'])
                st.markdown(content)
            
            with col2:
                st.markdown("### Article Details")
                st.markdown(f"**File:** {result['filename']}")
                st.markdown(f"**URL:** {result['url']}")
            
            st.markdown("---")

def main():
    st.title("SharePoint KB Assistant")
    
    st.markdown("""
    This application helps support agents by:
    1. Finding relevant KB articles from SharePoint based on incident details
    2. Posting the complete KB article content to the incident's work notes
    """)
    
    # Initialize session state for configuration
    if 'config' not in st.session_state:
        st.session_state.config = {
            'sharepoint': {
                'tenant_id': '',
                'client_id': '',
                'client_secret': '',
                'site_url': '',
                'library_name': ''
            },
            'servicenow': {
                'instance_url': '',
                'username': '',
                'password': ''
            }
        }
    
    # Initialize other session state variables
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'current_incident' not in st.session_state:
        st.session_state.current_incident = None
    if 'sharepoint_connected' not in st.session_state:
        st.session_state.sharepoint_connected = False
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    
    # Sidebar Configuration
    st.sidebar.header("Configuration")
    
    # SharePoint Settings
    with st.sidebar.expander("SharePoint Settings", expanded=True):
        st.session_state.config['sharepoint']['tenant_id'] = st.text_input(
            "Azure Tenant ID",
            value=st.session_state.config['sharepoint']['tenant_id']
        )
        st.session_state.config['sharepoint']['client_id'] = st.text_input(
            "Azure App Client ID",
            value=st.session_state.config['sharepoint']['client_id']
        )
        st.session_state.config['sharepoint']['client_secret'] = st.text_input(
            "Azure App Client Secret",
            type="password",
            value=st.session_state.config['sharepoint']['client_secret']
        )
        st.session_state.config['sharepoint']['site_url'] = st.text_input(
            "SharePoint Site URL",
            value=st.session_state.config['sharepoint']['site_url']
        )
        st.session_state.config['sharepoint']['library_name'] = st.text_input(
            "Document Library Name",
            value=st.session_state.config['sharepoint']['library_name']
        )
    
    # ServiceNow Settings
    with st.sidebar.expander("ServiceNow Settings", expanded=True):
        st.session_state.config['servicenow']['instance_url'] = st.text_input(
            "ServiceNow Instance URL",
            value=st.session_state.config['servicenow']['instance_url']
        )
        st.session_state.config['servicenow']['username'] = st.text_input(
            "ServiceNow Username",
            value=st.session_state.config['servicenow']['username']
        )
        st.session_state.config['servicenow']['password'] = st.text_input(
            "ServiceNow Password",
            type="password",
            value=st.session_state.config['servicenow']['password']
        )
    
    # Connect and Load KB button
    if st.sidebar.button("Connect & Load KB"):
        sharepoint_config = st.session_state.config['sharepoint']
        
        # Validate SharePoint settings
        if not all([
            sharepoint_config['tenant_id'],
            sharepoint_config['client_id'],
            sharepoint_config['client_secret'],
            sharepoint_config['site_url'],
            sharepoint_config['library_name']
        ]):
            st.error("Please fill in all SharePoint settings")
            return
        
        with st.spinner("Connecting to SharePoint..."):
            sp_client = initialize_sharepoint_client(
                sharepoint_config['tenant_id'],
                sharepoint_config['client_id'],
                sharepoint_config['client_secret']
            )
            
            if sp_client:
                with st.spinner("Loading Knowledge Base articles..."):
                    sharepoint_docs = get_sharepoint_documents(
                        sp_client,
                        sharepoint_config['site_url'],
                        sharepoint_config['library_name']
                    )
                    
                    if sharepoint_docs:
                        documents = create_kb_documents(sharepoint_docs)
                        if documents:
                            vector_store = initialize_vector_store(documents)
                            if vector_store:
                                st.session_state.vector_store = vector_store
                                st.session_state.sharepoint_connected = True
                                st.success(f"Successfully loaded {len(documents)} KB articles")
                            else:
                                st.error("Failed to initialize vector store")
                    else:
                        st.error("No documents found in the specified SharePoint library")
    
    # Main Interface - only show if vector store is initialized
    if st.session_state.vector_store:
        # Create tabs for different functionalities
        incident_tab, search_tab = st.tabs(["Incident Analysis", "Manual Search"])
        
        # Incident Analysis Tab
        with incident_tab:
            st.header("Incident KB Analysis")
            
            incident_number = st.text_input(
                "Enter Incident Number",
                help="e.g., INC0012345"
            )
            
            if incident_number and st.button("Analyze Incident"):
                servicenow_config = st.session_state.config['servicenow']
                
                # Validate ServiceNow settings
                if not all([
                    servicenow_config['instance_url'],
                    servicenow_config['username'],
                    servicenow_config['password']
                ]):
                    st.error("Please fill in all ServiceNow settings")
                    return
                
                with st.spinner("Fetching incident details..."):
                    snow_client = initialize_snow_client(
                        servicenow_config['instance_url'],
                        servicenow_config['username'],
                        servicenow_config['password']
                    )
                    
                    if snow_client:
                        incident = get_incident_details(snow_client, incident_number)
                        if incident:
                            st.session_state.current_incident = incident
                            
                            # Display incident details
                            display_incident_details(incident)
                            
                            # Search for relevant articles
                            search_text = create_search_query_from_incident(incident)
                            results = search_articles(st.session_state.vector_store, search_text)
                            
                            if results:
                                st.session_state.search_results = results
                                st.subheader(f"Found {len(results)} Relevant Articles")
                                display_results(results)
                                
                                # Store top result for work note posting
                                st.session_state.top_article = results[0]
                            else:
                                st.warning("No relevant articles found for this incident")
            
            # Work Notes Section
            if hasattr(st.session_state, 'top_article') and st.session_state.current_incident:
                st.markdown("---")
                st.subheader("Update Incident Work Notes")
                
                article = st.session_state.top_article
                
                with st.expander("Preview Work Note Content", expanded=True):
                    st.markdown(f"**KB Article**: {article['title']}")
                    st.markdown(f"**Filename**: {article['filename']}")
                    st.markdown(f"**URL**: {article['url']}")
                    st.markdown(f"**Relevance Score**: {article['similarity']:.2f}")
                    st.markdown("**Article Content**:")
                    st.markdown(clean_html_content(article['content']))
                
                if st.button("Post to Work Notes"):
                    servicenow_config = st.session_state.config['servicenow']
                    
                    with st.spinner("Updating incident..."):
                        snow_client = initialize_snow_client(
                            servicenow_config['instance_url'],
                            servicenow_config['username'],
                            servicenow_config['password']
                        )
                        
                        if snow_client:
                            success = post_work_note(
                                snow_client,
                                st.session_state.current_incident['sys_id'],
                                article
                            )
                            
                            if success:
                                st.success("Successfully updated work notes!")
                                del st.session_state.top_article
        
        # Manual Search Tab
        with search_tab:
            st.header("Manual KB Search")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                search_query = st.text_input(
                    "Search KB Articles",
                    help="Enter keywords to search KB articles"
                )
            
            with col2:
                n_results = st.number_input(
                    "Number of results",
                    min_value=1,
                    max_value=10,
                    value=5
                )
            
            if search_query and st.button("Search", key="manual_search"):
                results = search_articles(
                    st.session_state.vector_store,
                    search_query,
                    n_results
                )
                
                if results:
                    st.subheader(f"Found {len(results)} Relevant Articles")
                    display_results(results)
                else:
                    st.warning("No articles found matching your search")

if __name__ == "__main__":
    main()
