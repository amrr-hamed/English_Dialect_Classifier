import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import os
from pathlib import Path
import tempfile

# Import your existing modules
try:
    from audio_extractor import prepare_audio
    from dialect_predector import analyze_video_accent
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🎤 English Accent Analyzer",
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
    """Create confidence score chart for 1-minute chunks"""
    if not chunk_results:
        return None
    
    chunk_data = []
    for i, result in enumerate(chunk_results):
        chunk_data.append({
            'Minute': f"Min {i+1}",
            'Confidence': result['confidence'],
            'Accent': result['accent'],
            'Is Confident': '✓ High Confidence' if result['is_confident'] else '✗ Low Confidence'
        })
    
    df = pd.DataFrame(chunk_data)
    
    fig = px.bar(df, 
                 x='Minute', 
                 y='Confidence', 
                 color='Is Confident',
                 hover_data=['Accent'],
                 title='Confidence Scores by Minute',
                 color_discrete_map={'✓ High Confidence': '#28a745', '✗ Low Confidence': '#dc3545'})
    
    fig.update_layout(
        xaxis_title="Time Segment",
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
            label="🎯 Overall Confidence",
            value=f"{results['confidence_score']:.1%}",
            help="Overall confidence in the prediction"
        )
    
    with col2:
        st.metric(
            label="📊 Minutes Analyzed",
            value=f"{results['processed_chunks_count']} min",
            delta=f"of {results.get('duration_minutes', 0):.1f} min total"
        )
    
    with col3:
        st.metric(
            label="✅ High Confidence Segments",
            value=results['confident_chunks_count'],
            delta=f"{(results['confident_chunks_count']/results['processed_chunks_count']*100):.0f}%" if results['processed_chunks_count'] > 0 else "0%"
        )
    
    with col4:
        st.metric(
            label="⏱️ Processing Time",
            value=f"{results['processing_time']:.1f}s",
            help="Time taken to analyze the audio"
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
            "High Confidence Predictions"
        )
        if confident_chart:
            st.plotly_chart(confident_chart, use_container_width=True)
    
    # Detailed results table
    with st.expander("🔍 View Minute-by-Minute Results"):
        if results['chunk_results']:
            chunk_df = pd.DataFrame(results['chunk_results'])
            chunk_df.index = [f"Minute {i+1}" for i in range(len(chunk_df))]
            st.dataframe(chunk_df, use_container_width=True)
    
    # Summary statistics
    with st.expander("📋 Summary Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**High Confidence Predictions:**")
            if results['confident_accent_counts']:
                for accent, count in results['confident_accent_counts'].items():
                    percentage = (count / results['confident_chunks_count']) * 100
                    st.write(f"• {accent}: {count} segments ({percentage:.1f}%)")
            else:
                st.write("No high confidence predictions")
        
        with col2:
            st.write("**All Predictions:**")
            if results['all_accent_counts']:
                for accent, count in results['all_accent_counts'].items():
                    percentage = (count / results['processed_chunks_count']) * 100
                    st.write(f"• {accent}: {count} segments ({percentage:.1f}%)")

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎤 English Accent Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("Analyze English accents from video files, Loom videos, or direct media URLs. Audio is processed in 1-minute segments for detailed analysis.")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        step=0.05,
        help="Only predictions above this threshold are considered high confidence"
    )
    
    # Input section
    st.header("📥 Input Source")
    
    input_method = st.radio(
        "Choose input method:",
        ["URL (Loom or Direct Link)", "Upload File"],
        horizontal=True
    )
    
    source = None
    
    if input_method == "URL (Loom or Direct Link)":
        source = st.text_input(
            "Enter video URL:",
            placeholder="https://www.loom.com/share/...",
            help="Supports Loom videos and direct media URLs"
        )
        
        # URL examples
        with st.expander("🔗 Supported URL Examples"):
            st.write("• **Loom:** `https://www.loom.com/share/VIDEO_ID`")
            st.write("• **Direct MP4:** `https://example.com/video.mp4`")
            st.write("• **Direct audio:** `https://example.com/audio.mp3`")
            st.markdown('<div class="info-box">📝 <strong>Note:</strong> YouTube URLs are not supported to avoid authentication issues in deployment.</div>', unsafe_allow_html=True)
    
    else:  # Upload File
        uploaded_file = st.file_uploader(
            "Choose a video or audio file",
            type=['mp4', 'webm', 'avi', 'mov', 'mkv', 'm4v', 'mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac'],
            help="Upload video or audio files for accent analysis"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            with st.spinner("Saving uploaded file..."):
                source = save_uploaded_file(uploaded_file)
            
            if source:
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                file_size = len(uploaded_file.getbuffer()) / 1024 / 1024
                st.info(f"📊 File size: {file_size:.1f}MB")
            else:
                st.error("❌ Failed to save uploaded file")
    
    # Analysis button
    analyze_button = st.button(
        "🚀 Start Accent Analysis",
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
            progress_bar.progress(25)
            
            status_text.text("🧩 Creating 1-minute segments...")
            progress_bar.progress(50)
            
            status_text.text("🧠 Analyzing accent patterns...")
            progress_bar.progress(75)
            
            # Run analysis with the confidence threshold
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
        st.header("📊 Analysis Results")
        display_results(st.session_state.analysis_results)
    
    # Information section
    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        **English Accent Analyzer** uses advanced machine learning models to identify English accents from speech.
        
        **Key Features:**
        - 🎯 **1-minute segments:** Audio is processed in 1-minute chunks for detailed analysis
        - 🎤 **Accent detection:** Identifies British, American, Australian, and other English accents
        - 📊 **Confidence scoring:** Provides reliability scores for each prediction
        - 🔗 **Multiple sources:** Supports Loom videos, direct URLs, and file uploads
        
        **Supported Formats:**
        - **Video:** MP4, WebM, AVI, MOV, MKV, M4V
        - **Audio:** MP3, WAV, M4A, AAC, OGG, FLAC
        - **URLs:** Loom videos, direct media links
        
        **How it works:**
        1. Audio is extracted from your source
        2. Audio is split into 1-minute segments
        3. Each segment is analyzed for accent characteristics
        4. Results are combined with confidence weighting
        5. Final accent prediction is provided
        
        **Best Results:**
        - Use clear speech audio
        - Longer videos provide more accurate results
        - Multiple speakers may affect accuracy
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 **Deployment Ready:** Optimized for Hugging Face Spaces deployment")

if __name__ == "__main__":
    main()