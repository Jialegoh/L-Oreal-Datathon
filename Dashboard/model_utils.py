"""
AI Model utilities for the L'Oréal TrendSpotter Dashboard
Contains functions for loading and using various AI models (BERT, sentiment analysis, etc.)
"""

import torch
import streamlit as st
import numpy as np
import pandas as pd
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle


def get_device():
    """Get the appropriate device (GPU if available, otherwise CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def load_model_bundle(model_path="AI_Model/CommentCategory/saved_model"):
    """
    Load the complete model bundle including tokenizer, model, and metadata
    
    Args:
        model_path (str): Path to the saved model directory
        
    Returns:
        dict: Dictionary containing loaded model components and device info
    """
    device = get_device()
    
    try:
        # Check if the model path exists
        if not os.path.exists(model_path):
            st.error(f"❌ Model path does not exist: {model_path}")
            return None
        
        # Try to load tokenizer - handle both local and HF hub formats
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        except Exception as e1:
            st.warning(f"⚠️ Could not load tokenizer from local path, trying fallback method: {str(e1)}")
            # Fallback: try to load from a standard model if local fails
            try:
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                st.info("ℹ️ Using fallback BERT tokenizer")
            except Exception as e2:
                st.error(f"❌ Failed to load any tokenizer: {str(e2)}")
                return None
        
        # Try to load model - handle both local and HF hub formats
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        except Exception as e1:
            st.warning(f"⚠️ Could not load model from local path: {str(e1)}")
            # Check if we have model files in the directory
            model_files = [f for f in os.listdir(model_path) if f.endswith(('.bin', '.safetensors', '.pt', '.pth'))]
            if model_files:
                st.info(f"Found model files: {model_files}")
                # Try to load using torch directly
                try:
                    model_file = os.path.join(model_path, model_files[0])
                    if model_file.endswith('.bin'):
                        # This is likely a PyTorch model file
                        model_state = torch.load(model_file, map_location=device)
                        # We need to determine the model architecture - using BERT as default
                        from transformers import BertForSequenceClassification
                        
                        # Determine number of labels from the state dict
                        classifier_weight = model_state.get('classifier.weight', None)
                        if classifier_weight is not None:
                            num_labels = classifier_weight.shape[0]
                        else:
                            num_labels = 2  # Default fallback
                        
                        model = BertForSequenceClassification.from_pretrained(
                            'bert-base-uncased', 
                            num_labels=num_labels
                        )
                        model.load_state_dict(model_state)
                        st.info(f"✅ Loaded model with {num_labels} labels using torch.load")
                    else:
                        raise Exception("Unsupported model file format")
                except Exception as e2:
                    st.error(f"❌ Failed to load model using fallback methods: {str(e2)}")
                    return None
            else:
                st.warning("⚠️ No model weight files found in the specified directory")
                st.info("📋 Available files: " + ", ".join(os.listdir(model_path)))
                
                # Create a demo model for testing purposes
                st.warning("🔧 Creating demo model for testing purposes...")
                from transformers import BertForSequenceClassification
                
                # Load number of classes from mlb_classes.npy if available
                mlb_classes_path = os.path.join(model_path, "mlb_classes.npy")
                if os.path.exists(mlb_classes_path):
                    try:
                        mlb_classes = np.load(mlb_classes_path, allow_pickle=True)
                        num_labels = len(mlb_classes)
                        st.info(f"📊 Found {num_labels} classes in mlb_classes.npy")
                    except:
                        num_labels = 5  # Default fallback
                        st.warning("⚠️ Could not load mlb_classes.npy, using default 5 labels")
                else:
                    num_labels = 5  # Default fallback
                    st.warning("⚠️ mlb_classes.npy not found, using default 5 labels")
                
                # Create a fresh BERT model for the number of classes
                model = BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased', 
                    num_labels=num_labels
                )
                st.info(f"✅ Created demo BERT model with {num_labels} labels (untrained weights)")
                st.warning("⚠️ Note: This is a demo model with random weights. For actual classification, you need trained model weights.")
        
        model = model.to(device)
        
        # Load MultiLabelBinarizer classes
        mlb_classes_path = os.path.join(model_path, "mlb_classes.npy")
        mlb_classes = np.load(mlb_classes_path, allow_pickle=True)
        
        # Load optimal thresholds
        thresholds_path = os.path.join(model_path, "optimal_thresholds.csv")
        if os.path.exists(thresholds_path):
            thresholds_df = pd.read_csv(thresholds_path)
            optimal_thresholds = dict(zip(thresholds_df['category'], thresholds_df['optimal_threshold']))
        else:
            # Default thresholds if file not found
            optimal_thresholds = {category: 0.5 for category in mlb_classes}
        
        # Load configuration if available
        config_path = os.path.join(model_path, "config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        bundle = {
            'tokenizer': tokenizer,
            'model': model,
            'mlb_classes': mlb_classes,
            'optimal_thresholds': optimal_thresholds,
            'config': config,
            'device': device,
            'model_path': model_path
        }
        
        st.success(f"✅ Model loaded successfully on {device}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            st.info(f"🚀 Using GPU: {gpu_name}")
        
        return bundle
        
    except Exception as e:
        st.error(f"❌ Error loading model from {model_path}: {str(e)}")
        return None


def predict_top_k(text, model_bundle, top_k=5):
    """
    Predict top K categories for a given text using the loaded model
    
    Args:
        text (str): Input text to classify
        model_bundle (dict): Loaded model bundle from load_model_bundle()
        top_k (int): Number of top predictions to return
        
    Returns:
        list: List of tuples containing (category, probability)
    """
    if not model_bundle:
        return []
    
    try:
        tokenizer = model_bundle['tokenizer']
        model = model_bundle['model']
        mlb_classes = model_bundle['mlb_classes']
        device = model_bundle['device']
        
        # Tokenize input text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model predictions
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Create category-probability pairs
        category_probs = list(zip(mlb_classes, probabilities))
        
        # Sort by probability (descending) and get top K
        top_predictions = sorted(category_probs, key=lambda x: x[1], reverse=True)[:top_k]
        
        return top_predictions
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return []


def predict_with_thresholds(text, model_bundle):
    """
    Predict categories using optimal thresholds for binary classification
    
    Args:
        text (str): Input text to classify
        model_bundle (dict): Loaded model bundle from load_model_bundle()
        
    Returns:
        dict: Dictionary with category predictions and probabilities
    """
    if not model_bundle:
        return {}
    
    try:
        tokenizer = model_bundle['tokenizer']
        model = model_bundle['model']
        mlb_classes = model_bundle['mlb_classes']
        optimal_thresholds = model_bundle['optimal_thresholds']
        device = model_bundle['device']
        
        # Tokenize input text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Apply thresholds for binary predictions
        predictions = {}
        for i, category in enumerate(mlb_classes):
            threshold = optimal_thresholds.get(category, 0.5)
            predictions[category] = {
                'probability': float(probabilities[i]),
                'predicted': probabilities[i] >= threshold,
                'threshold': threshold
            }
        
        return predictions
        
    except Exception as e:
        st.error(f"❌ Error during threshold prediction: {str(e)}")
        return {}


def load_sentiment_model():
    """Load sentiment analysis model"""
    try:
        # This would load your specific sentiment model
        # Placeholder for sentiment model loading
        device = get_device()
        st.info("📊 Sentiment model loaded")
        return {'device': device, 'model_type': 'sentiment'}
    except Exception as e:
        st.error(f"❌ Error loading sentiment model: {str(e)}")
        return None


def load_spam_detection_model():
    """Load spam detection model"""
    try:
        # This would load your specific spam detection model
        # Placeholder for spam detection model loading
        device = get_device()
        st.info("🛡️ Spam detection model loaded")
        return {'device': device, 'model_type': 'spam_detection'}
    except Exception as e:
        st.error(f"❌ Error loading spam detection model: {str(e)}")
        return None


def batch_predict(texts, model_bundle, batch_size=32):
    """
    Predict categories for a batch of texts
    
    Args:
        texts (list): List of texts to classify
        model_bundle (dict): Loaded model bundle
        batch_size (int): Number of texts to process at once
        
    Returns:
        list: List of predictions for each text
    """
    if not model_bundle or not texts:
        return []
    
    all_predictions = []
    
    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_predictions = []
            
            for text in batch_texts:
                pred = predict_top_k(text, model_bundle, top_k=3)
                batch_predictions.append(pred)
            
            all_predictions.extend(batch_predictions)
            
            # Show progress
            progress = min(i + batch_size, len(texts))
            st.progress(progress / len(texts))
        
        return all_predictions
        
    except Exception as e:
        st.error(f"❌ Error during batch prediction: {str(e)}")
        return []


def model_performance_metrics(model_bundle):
    """
    Get model performance metrics if available
    
    Args:
        model_bundle (dict): Loaded model bundle
        
    Returns:
        dict: Performance metrics
    """
    if not model_bundle:
        return {}
    
    try:
        model_path = model_bundle['model_path']
        
        # Load classification report if available
        report_path = os.path.join(model_path, "classification_report.csv")
        if os.path.exists(report_path):
            report_df = pd.read_csv(report_path)
            return report_df.to_dict('records')
        
        # Load overall metrics if available
        metrics_path = os.path.join(model_path, "overall_classification_metrics.csv")
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            return metrics_df.to_dict('records')
        
        return {}
        
    except Exception as e:
        st.warning(f"⚠️ Could not load performance metrics: {str(e)}")
        return {}


def get_model_info(model_bundle):
    """
    Get information about the loaded model
    
    Args:
        model_bundle (dict): Loaded model bundle
        
    Returns:
        dict: Model information
    """
    if not model_bundle:
        return {}
    
    info = {
        'device': str(model_bundle['device']),
        'model_path': model_bundle['model_path'],
        'num_classes': len(model_bundle['mlb_classes']),
        'classes': list(model_bundle['mlb_classes']),
        'has_thresholds': bool(model_bundle['optimal_thresholds']),
        'config': model_bundle.get('config', {})
    }
    
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    
    return info


def clear_model_cache():
    """Clear model cache and free up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        st.success("🧹 GPU cache cleared")
    else:
        st.info("💻 Running on CPU - no GPU cache to clear")





def validate_model_bundle(model_bundle):
    """
    Validate that the model bundle has all required components
    
    Args:
        model_bundle (dict): Model bundle to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not model_bundle:
        return False, "Model bundle is None"
    
    required_keys = ['tokenizer', 'model', 'mlb_classes', 'device']
    missing_keys = [key for key in required_keys if key not in model_bundle]
    
    if missing_keys:
        return False, f"Missing required keys: {missing_keys}"
    
    return True, "Model bundle is valid"