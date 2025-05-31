import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import re
from datetime import datetime
import numpy as np
from dialect_predector import analyze_video_accent

# Import your accent analysis function
# from your_accent_module import analyze_video_accent

# Page configuration
st.set_page_config(
    page_title="🎤 AI Accent Analyzer",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .analysis-section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e0e6ed;
    }
    
    .accent-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .accent-primary {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
    }
    
    .accent-secondary {
        background: linear-gradient(45deg, #ffecd2, #fcb69f);
        color: #333;
    }
    
    .processing-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48cae4, #06ffa5);
        height: 20px;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .chunk-result {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 5px;
    }
    
    .chunk-result.low-confidence {
        border-left-color: #ffc107;
    }
    
    .sidebar-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def validate_url(url):
    """Validate if the URL is a valid YouTube URL"""
    youtube_patterns = [
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/',
        r'(https?://)?(www\.)?youtube\.com/shorts/',
        r'(https?://)?(www\.)?youtu\.be/'
    ]
    
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True
    return False

def create_confidence_gauge(confidence):
    """Create a beautiful confidence gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score"},
        delta = {'reference': 70},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_accent_distribution_chart(accent_counts, title="Accent Distribution"):
    """Create a beautiful pie chart for accent distribution"""
    if not accent_counts:
        return None
        
    accents = list(accent_counts.keys())
    counts = list(accent_counts.values())
    
    fig = px.pie(
        values=counts, 
        names=accents,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=12)
    )
    
    return fig

def create_chunk_confidence_chart(chunk_results):
    """Create a chart showing confidence over chunks"""
    if not chunk_results:
        return None
        
    df = pd.DataFrame(chunk_results)
    
    fig = px.line(
        df, 
        x='chunk', 
        y='confidence',
        title='Confidence Score Across Audio Chunks',
        markers=True,
        color='accent',
        hover_data=['accent', 'is_confident']
    )
    
    fig.add_hline(y=0.6, line_dash="dash", line_color="red", 
                  annotation_text="Confidence Threshold (60%)")
    
    fig.update_layout(
        height=400,
        xaxis_title="Chunk Number",
        yaxis_title="Confidence Score",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_detailed_analysis(result):
    """Create detailed analysis section"""
    if not result or not result.get("success"):
        return
    
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown("## 📊 Detailed Analysis")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "🎯 Final Accent", 
            result['predicted_accent'],
            f"{result['confidence_percentage']}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "📦 Chunks Processed", 
            f"{result['processed_chunks_count']}/{result['available_chunks_count']}",
            f"Confident: {result.get('confident_chunks_count', 0)}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "⏱️ Processing Time", 
            f"{result['processing_time']:.1f}s",
            f"Audio: {result.get('duration_minutes', 0):.1f}min" if result.get('duration_minutes') else ""
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        early_stopped_text = "Yes ⚡" if result.get('early_stopped') else "No 🔄"
        st.metric(
            "🛑 Early Stopped", 
            early_stopped_text,
            f"Threshold: {result.get('confidence_threshold', 0.6)*100:.0f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence gauge
        gauge_fig = create_confidence_gauge(result['confidence_score'])
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Accent distribution (confident predictions)
        if result.get('confident_accent_counts'):
            pie_fig = create_accent_distribution_chart(
                result['confident_accent_counts'], 
                "Confident Predictions Distribution"
            )
            if pie_fig:
                st.plotly_chart(pie_fig, use_container_width=True)
    
    with col2:
        # Chunk confidence over time
        if result.get('chunk_results'):
            confidence_fig = create_chunk_confidence_chart(result['chunk_results'])
            if confidence_fig:
                st.plotly_chart(confidence_fig, use_container_width=True)
        
        # All predictions distribution
        if result.get('all_accent_counts') and len(result['all_accent_counts']) > 1:
            all_pie_fig = create_accent_distribution_chart(
                result['all_accent_counts'], 
                "All Predictions Distribution"
            )
            if all_pie_fig:
                st.plotly_chart(all_pie_fig, use_container_width=True)

def display_chunk_details(chunk_results, confidence_threshold=0.6):
    """Display detailed chunk-by-chunk results"""
    if not chunk_results:
        return
    
    st.markdown("### 🔍 Chunk-by-Chunk Analysis")
    
    # Summary statistics
    confident_chunks = [r for r in chunk_results if r.get('is_confident', r['confidence'] > confidence_threshold)]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Total Chunks:** {len(chunk_results)}")
    with col2:
        st.success(f"**Confident Chunks:** {len(confident_chunks)}")
    with col3:
        confidence_rate = len(confident_chunks) / len(chunk_results) * 100 if chunk_results else 0
        st.warning(f"**Confidence Rate:** {confidence_rate:.1f}%")
    
    # Detailed results
    with st.expander("📋 View Detailed Chunk Results", expanded=False):
        for i, result in enumerate(chunk_results):
            confidence = result['confidence']
            is_confident = result.get('is_confident', confidence > confidence_threshold)
            
            confidence_emoji = "✅" if is_confident else "⚠️"
            confidence_class = "" if is_confident else "low-confidence"
            
            st.markdown(f"""
            <div class="chunk-result {confidence_class}">
                <strong>Chunk {result['chunk']}</strong> {confidence_emoji}<br>
                <strong>Accent:</strong> {result['accent']}<br>
                <strong>Confidence:</strong> {confidence:.3f} ({confidence*100:.1f}%)<br>
                <strong>Status:</strong> {'Confident' if is_confident else 'Low Confidence'}
            </div>
            """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎤 AI Accent Analyzer</h1>
        <p>Analyze accents from YouTube videos using advanced AI models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-info">
            <h3>🔧 Configuration</h3>
            <p>Adjust analysis parameters</p>
        </div>
        """, unsafe_allow_html=True)
        
        confidence_threshold = st.slider(
            "🎯 Confidence Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.6, 
            step=0.05,
            help="Only predictions above this confidence level are considered reliable"
        )
        
        early_stopping_threshold = st.slider(
            "⚡ Early Stopping Threshold", 
            min_value=2, 
            max_value=10, 
            value=3,
            help="Stop processing after this many consecutive confident predictions"
        )
        
        st.markdown("---")
        
        st.markdown("""
        ### 📋 Supported Formats
        - YouTube videos
        - YouTube Shorts
        - YouTube Music
        - Youtu.be links
        
        ### ⚙️ How it works
        1. **Audio Extraction**: Extracts audio from video
        2. **Chunking**: Splits audio into manageable segments
        3. **AI Analysis**: Uses SpeechBrain model for accent detection
        4. **Confidence Filtering**: Only considers high-confidence predictions
        5. **Results**: Provides detailed analysis and visualization
        """)
    
    # Main interface
    st.markdown("## 🔗 Enter Video URL")
    
    # URL input with examples
    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=example or https://youtu.be/example",
            help="Paste any YouTube video URL here"
        )
    
    with col2:
        st.markdown("**Quick Examples:**")
        example_urls = [
            "https://www.youtube.com/shorts/mxMzNp3RfpA",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=example"
        ]
        
        for i, url in enumerate(example_urls):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state.example_url = url
                st.rerun()
    
    # Use example URL if selected
    if hasattr(st.session_state, 'example_url'):
        video_url = st.session_state.example_url
        delattr(st.session_state, 'example_url')
    
    # URL validation
    if video_url:
        if validate_url(video_url):
            st.success("✅ Valid YouTube URL detected!")
        else:
            st.error("❌ Please enter a valid YouTube URL")
            st.stop()
    
    # Analysis button
    if st.button("🚀 Analyze Accent", type="primary", disabled=not video_url):
        if not video_url:
            st.warning("Please enter a video URL first!")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Simulate the analysis process with progress updates
            status_text.text("🔄 Initializing analysis...")
            progress_bar.progress(10)
            time.sleep(1)
            
            status_text.text("🎵 Extracting audio from video...")
            progress_bar.progress(30)
            time.sleep(1)
            
            status_text.text("🧠 Loading AI model...")
            progress_bar.progress(50)
            time.sleep(1)
            
            status_text.text("🔍 Analyzing accent patterns...")
            progress_bar.progress(80)
            
            # Here you would call your actual analysis function
            # result = analyze_video_accent(video_url, confidence_threshold)
            
            # For demo purposes, creating mock result
            result = analyze_video_accent(video_url, confidence_threshold)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            if result["success"]:
                st.success("🎉 Analysis completed successfully!")
                
                # Main result highlight
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
                    <h2>🎤 Detected Accent: {result['predicted_accent']}</h2>
                    <h3>📊 Confidence: {result['confidence_percentage']}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed analysis
                create_detailed_analysis(result)
                
                # Chunk details
                if result.get('chunk_results'):
                    display_chunk_details(result['chunk_results'], confidence_threshold)
                
                # Raw data download
                with st.expander("📥 Download Results", expanded=False):
                    # Convert results to DataFrame for download
                    if result.get('chunk_results'):
                        df = pd.DataFrame(result['chunk_results'])
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download Chunk Results (CSV)",
                            data=csv,
                            file_name=f"accent_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    # JSON download
                    import json
                    json_str = json.dumps(result, indent=2, default=str)
                    st.download_button(
                        label="📋 Download Full Results (JSON)",
                        data=json_str,
                        file_name=f"accent_analysis_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            else:
                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ An error occurred during analysis: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🎤 AI Accent Analyzer | Built with Streamlit & SpeechBrain</p>
        <p>Analyze accents from YouTube videos with confidence-based filtering</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()