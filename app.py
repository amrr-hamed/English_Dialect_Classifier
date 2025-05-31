import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from pathlib import Path
import tempfile
import shutil

# Import your existing modules
try:
    from audio_extractor import prepare_audio
    from dialect_predector import analyze_video_accent
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🎤 Accent Analyzer",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory"""
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

def create_confidence_chart(chunk_results):
    """Create confidence score chart for chunks"""
    if not chunk_results:
        return None
    
    chunk_data = []
    for result in chunk_results:
        chunk_data.append({
            'Chunk': result['chunk'],
            'Confidence': result['confidence'],
            'Accent': result['accent'],
            'Is Confident': '✓ Confident' if result['is_confident'] else '✗ Low Confidence'
        })
    
    df = pd.DataFrame(chunk_data)
    
    fig = px.bar(df, 
                 x='Chunk', 
                 y='Confidence', 
                 color='Is Confident',
                 hover_data=['Accent'],
                 title='Confidence Scores by Chunk',
                 color_discrete_map={'✓ Confident': '#28a745', '✗ Low Confidence': '#dc3545'})
    
    fig.update_layout(
        xaxis_title="Chunk Number",
        yaxis_title="Confidence Score",
        showlegend=True,
        height=400
    )
    
    return fig

def create_accent_distribution_chart(accent_counts, title="Accent Distribution"):
    """Create pie chart for accent distribution"""
    if not accent_counts:
        return None
    
    accents = list(accent_counts.keys())
    counts = list(accent_counts.values())
    
    fig = px.pie(values=counts, 
                 names=accents, 
                 title=title,
                 color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def display_results(results):
    """Display analysis results with charts and metrics"""
    if not results['success']:
        st.markdown(f'<div class="error-box">❌ <strong>Error:</strong> {results["error"]}</div>', 
                   unsafe_allow_html=True)
        return
    
    # Main result
    st.markdown(f'<div class="success-box">🎤 <strong>Detected Accent:</strong> {results["predicted_accent"]}</div>', 
               unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Confidence Score",
            value=f"{results['confidence_score']:.3f}",
            delta=f"{results['confidence_percentage']}"
        )
    
    with col2:
        st.metric(
            label="📊 Chunks Processed",
            value=f"{results['processed_chunks_count']}/{results['available_chunks_count']}",
            delta="Early stopped" if results.get('early_stopped', False) else "Complete"
        )
    
    with col3:
        st.metric(
            label="✅ Confident Predictions",
            value=results['confident_chunks_count'],
            delta=f"{(results['confident_chunks_count']/results['processed_chunks_count']*100):.1f}%"
        )
    
    with col4:
        st.metric(
            label="⏱️ Processing Time",
            value=f"{results['processing_time']:.1f}s",
            delta=f"{results.get('duration_minutes', 0):.1f}min video"
        )
    
    # Detailed Analysis
    st.subheader("📈 Detailed Analysis")
    
    # Create two columns for charts
    chart_col1, chart_col2 = st.columns(2)
    
    # Confidence chart
    with chart_col1:
        confidence_chart = create_confidence_chart(results['chunk_results'])
        if confidence_chart:
            st.plotly_chart(confidence_chart, use_container_width=True)
    
    # Accent distribution for confident predictions
    with chart_col2:
        confident_chart = create_accent_distribution_chart(
            results['confident_accent_counts'], 
            "Confident Predictions Distribution"
        )
        if confident_chart:
            st.plotly_chart(confident_chart, use_container_width=True)
    
    # All predictions distribution
    if results['all_accent_counts'] != results['confident_accent_counts']:
        st.subheader("📊 All Predictions (Including Low Confidence)")
        all_chart = create_accent_distribution_chart(
            results['all_accent_counts'], 
            "All Predictions Distribution"
        )
        if all_chart:
            st.plotly_chart(all_chart, use_container_width=True)
    
    # Detailed chunk results table
    with st.expander("🔍 View Detailed Chunk Results"):
        chunk_df = pd.DataFrame(results['chunk_results'])
        st.dataframe(chunk_df, use_container_width=True)
    
    # Summary statistics
    with st.expander("📋 Summary Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Confident Predictions:**")
            for accent, count in results['confident_accent_counts'].items():
                percentage = (count / results['confident_chunks_count']) * 100
                st.write(f"• {accent}: {count} chunks ({percentage:.1f}%)")
        
        with col2:
            st.write("**All Predictions:**")
            for accent, count in results['all_accent_counts'].items():
                percentage = (count / results['processed_chunks_count']) * 100
                st.write(f"• {accent}: {count} chunks ({percentage:.1f}%)")

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎤 Accent Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("Analyze accents from video files, URLs, or audio sources using advanced AI models.")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        step=0.05,
        help="Only predictions above this threshold are considered confident"
    )
    
    early_stopping = st.sidebar.checkbox(
        "Enable Early Stopping",
        value=True,
        help="Stop processing when 3 consecutive confident predictions agree"
    )
    
    # Input section
    st.header("📥 Input Source")
    
    input_method = st.radio(
        "Choose input method:",
        ["URL (YouTube, Loom, etc.)", "Upload File"],
        horizontal=True
    )
    
    source = None
    
    if input_method == "URL (YouTube, Loom, etc.)":
        source = st.text_input(
            "Enter video URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Supports YouTube, Loom, and direct media URLs"
        )
        
        # URL examples
        with st.expander("🔗 Supported URL Examples"):
            st.write("• YouTube: `https://www.youtube.com/watch?v=VIDEO_ID`")
            st.write("• YouTube Shorts: `https://www.youtube.com/shorts/VIDEO_ID`")
            st.write("• Loom: `https://www.loom.com/share/VIDEO_ID`")
            st.write("• Direct media files: `https://example.com/video.mp4`")
    
    else:  # Upload File
        uploaded_file = st.file_uploader(
            "Choose a video or audio file",
            type=['mp4', 'webm', 'avi', 'mov', 'mkv', 'm4v', '3gp', 'mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac'],
            help="Upload video or audio files for accent analysis"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            with st.spinner("Saving uploaded file..."):
                source = save_uploaded_file(uploaded_file)
                st.session_state.uploaded_file_path = source
            
            if source:
                st.success(f"✅ File uploaded: {uploaded_file.name}")
            else:
                st.error("❌ Failed to save uploaded file")
    
    # Analysis button
    analyze_button = st.button(
        "🚀 Start Analysis",
        type="primary",
        disabled=not source or st.session_state.processing,
        use_container_width=True
    )
    
    # Process analysis
    if analyze_button and source:
        st.session_state.processing = True
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🎵 Extracting audio...")
            progress_bar.progress(20)
            
            status_text.text("🧠 Loading AI model...")
            progress_bar.progress(40)
            
            status_text.text("🔍 Analyzing accent...")
            progress_bar.progress(60)
            
            # Run analysis
            results = analyze_video_accent(source, confidence_threshold=confidence_threshold)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Store results in session state
            st.session_state.analysis_results = results
            
            # Clean up progress indicators
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            progress_bar.empty()
            status_text.empty()
        
        finally:
            st.session_state.processing = False
    
    # Display results
    if st.session_state.analysis_results:
        st.header("📊 Results")
        display_results(st.session_state.analysis_results)
    
    # Information section
    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        **Accent Analyzer** uses advanced machine learning models to identify accents from speech in videos and audio files.
        
        **Features:**
        - Supports multiple input sources (URLs, file uploads)
        - Smart chunking for efficient processing
        - Confidence-based predictions
        - Early stopping for faster results
        - Detailed analysis with visualizations
        
        **Supported Formats:**
        - **Video:** MP4, WebM, AVI, MOV, MKV, M4V, 3GP
        - **Audio:** MP3, WAV, M4A, AAC, OGG, FLAC
        - **URLs:** YouTube, Loom, direct media links
        
        **How it works:**
        1. Audio is extracted from the source
        2. Audio is chunked into smaller segments
        3. Each chunk is analyzed for accent features
        4. Results are aggregated with confidence scoring
        5. Final prediction is made based on confident predictions
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and SpeechBrain")

if __name__ == "__main__":
    main()