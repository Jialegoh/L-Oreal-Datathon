"""
Classification Model Tab Module for L'Oréal TrendSpotter Dashboard
Handles the BERT-based multi-label classification model interface
"""

import streamlit as st
import pandas as pd
from model_utils import load_model_bundle, predict_top_k, get_device
from utils import display_device_info


def render_classification_tab():
    """Render the Classification Model tab with GPU-enabled inference"""
    st.markdown('<h3 style="color:#000;">Multi-Label Classification Model</h3>', unsafe_allow_html=True)
    
    # Display device information immediately
    device_type = display_device_info()
    
    # Load model
    if 'model_bundle' not in st.session_state:
        with st.spinner("Loading BERT classification model..."):
            st.session_state.model_bundle = load_model_bundle()
    
    model_bundle = st.session_state.model_bundle
    
    if not model_bundle:
        st.error("❌ Failed to load classification model. Please check the model files.")
        return
    
    st.success(f"✅ Model loaded successfully on {model_bundle['device']}")
    
    # Model information
    with st.expander("ℹ️ Model Information"):
        st.write(f"**Device**: {model_bundle['device']}")
        st.write(f"**Number of Categories**: {len(model_bundle['mlb_classes'])}")
        st.write(f"**Categories**: {', '.join(model_bundle['mlb_classes'][:10])}{'...' if len(model_bundle['mlb_classes']) > 10 else ''}")
    
    st.divider()
    
    # Text input for classification
    st.markdown('<h4 style="color:#000;">Classify Your Text</h4>', unsafe_allow_html=True)
    
    # Sample texts for quick testing
    sample_texts = [
        "This product is amazing! I love how it makes my skin feel so smooth and radiant.",
        "The packaging could be better, but the quality of the product is outstanding.",
        "Not worth the price. I've used better products for half the cost.",
        "Perfect for sensitive skin. No irritation at all and great results.",
        "The customer service was terrible, but the product works well."
    ]
    
    # Text input options
    input_method = st.radio("Choose input method:", ["Type your own text", "Select sample text"])
    
    if input_method == "Type your own text":
        user_text = st.text_area("Enter your text for classification:", 
                                height=100, 
                                placeholder="Type your comment here...")
    else:
        selected_sample = st.selectbox("Select a sample text:", sample_texts)
        user_text = selected_sample
        st.text_area("Selected text:", value=user_text, height=100, disabled=True)
    
    # Top K selection
    top_k = st.slider("Number of top predictions to show:", min_value=1, max_value=10, value=5)
    
    # Prediction
    if st.button("🚀 Classify Text", type="primary"):
        if user_text.strip():
            with st.spinner("Classifying text..."):
                predictions = predict_top_k(user_text, model_bundle, top_k=top_k)
            
            if predictions:
                st.markdown('<h4 style="color:#000;">📊 Classification Results</h4>', unsafe_allow_html=True)
                
                # Display results in a nice format
                results_df = pd.DataFrame(predictions, columns=['Category', 'Probability'])
                results_df['Probability'] = results_df['Probability'].apply(lambda x: f"{x:.4f}")
                results_df['Confidence'] = results_df['Probability'].astype(float).apply(
                    lambda x: "High" if x > 0.7 else "Medium" if x > 0.4 else "Low"
                )
                
                st.dataframe(results_df, use_container_width=True)
                
                # Visualization
                import plotly.express as px
                fig = px.bar(results_df, x='Category', y='Probability', 
                           title='Classification Probabilities')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ Failed to classify text. Please try again.")
        else:
            st.warning("⚠️ Please enter some text to classify.")
    
    st.divider()
    
    # Batch classification section
    st.markdown('<h4 style="color:#000;">📄 Batch Classification</h4>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a CSV file with texts to classify", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            text_column = st.selectbox("Select the column containing text:", df.columns)
            
            if st.button("🔄 Process Batch Classification"):
                if text_column in df.columns:
                    with st.spinner(f"Processing {len(df)} texts..."):
                        # Process in batches to avoid memory issues
                        batch_size = 50
                        all_results = []
                        
                        progress_bar = st.progress(0)
                        
                        for i in range(0, len(df), batch_size):
                            batch = df.iloc[i:i+batch_size]
                            batch_results = []
                            
                            for idx, row in batch.iterrows():
                                text = str(row[text_column])
                                predictions = predict_top_k(text, model_bundle, top_k=1)
                                if predictions:
                                    top_category = predictions[0][0]
                                    top_probability = predictions[0][1]
                                else:
                                    top_category = "Unknown"
                                    top_probability = 0.0
                                
                                batch_results.append({
                                    'Original_Text': text,
                                    'Predicted_Category': top_category,
                                    'Probability': top_probability
                                })
                            
                            all_results.extend(batch_results)
                            progress_bar.progress(min(i + batch_size, len(df)) / len(df))
                        
                        results_df = pd.DataFrame(all_results)
                        st.success(f"✅ Processed {len(results_df)} texts successfully!")
                        
                        # Display results
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="classification_results.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("❌ Selected column not found in the uploaded file.")
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")


def apply_brand_style(fig):
    """Apply L'Oréal brand styling to plotly figures"""
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black',
        title_font_color='black',
        title_font_size=16,
        font_size=12
    )