"""
AI-Powered Phishing Website Detection System
Main module that implements the core functionality for detecting phishing websites
using machine learning and feature engineering techniques.
"""

import os
import pandas as pd
import numpy as np
import joblib
import requests
from urllib.parse import urlparse
import whois
import socket
import ssl
import re
from datetime import datetime
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')
from data_processor import DataProcessor

class PhishingDetector:
    """
    A complete phishing website detection system using machine learning.
    This class handles data preprocessing, feature extraction, model training,
    evaluation, and prediction for phishing URL detection.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the PhishingDetector object.
        
        Args:
            model_path (str, optional): Path to a pre-trained model file.
        """
        self.features = []
        self.model = None
        self.scaler = None
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
            print(f"Model loaded from {model_path}")
    
    def _load_model(self, model_path):
        """Load a pre-trained model from disk."""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.features = model_data['features']
    
    def save_model(self, model_path):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model has been trained yet.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_dataset(self, filepath):
        """
        Load dataset from a CSV file containing URLs and their labels.
        
        Args:
            filepath (str): Path to the CSV file with columns 'url' and 'is_phishing'
            
        Returns:
            pd.DataFrame: The loaded dataset
        """
        df = pd.read_csv(filepath)
        print(f"Loaded dataset with {len(df)} entries")
        return df
    
    def extract_features(self, url, html_content=None):
        """
        Extract features from a URL and its HTML content.
        
        Args:
            url (str): The URL to analyze
            html_content (str, optional): The HTML content of the website
            
        Returns:
            dict: Dictionary of extracted features
        """
        features = {}
        
        # URL-based features
        features.update(self._extract_url_features(url))
        
        # Domain-based features
        features.update(self._extract_domain_features(url))
        
        # Content-based features (if HTML is provided or can be fetched)
        if html_content is None:
            try:
                response = requests.get(url, timeout=5, verify=False)
                html_content = response.text
            except:
                html_content = ""
        
        if html_content:
            features.update(self._extract_content_features(html_content, url))
        
        return features
    
    def _extract_url_features(self, url):
        """Extract features from the URL structure."""
        features = {}
        
        # Basic URL characteristics
        features['url_length'] = len(url)
        features['domain_length'] = len(urlparse(url).netloc)
        
        # Special character counts
        features['url_at_symbol'] = 1 if '@' in url else 0
        features['url_double_slash'] = url.count('//')
        features['url_dash_count'] = url.count('-')
        features['url_underscore_count'] = url.count('_')
        features['url_dot_count'] = url.count('.')
        features['url_percent_count'] = url.count('%')
        features['url_query_param_count'] = url.count('&') + 1 if '?' in url else 0
        
        # URL structure patterns
        parsed_url = urlparse(url)
        features['has_subdomain'] = 1 if len(parsed_url.netloc.split('.')) > 2 else 0
        features['has_https'] = 1 if parsed_url.scheme == "https" else 0
        features['path_length'] = len(parsed_url.path)
        features['hostname_length'] = len(parsed_url.netloc)
        
        # Advanced patterns
        features['has_ip_address'] = 1 if self._has_ip_address(url) else 0
        features['has_suspicious_words'] = self._check_suspicious_keywords(url)
        
        return features
    
    def _has_ip_address(self, url):
        """Check if the URL uses an IP address instead of domain name."""
        pattern = re.compile(r'(?:\d{1,3}\.){3}\d{1,3}')
        return 1 if pattern.search(urlparse(url).netloc) else 0
    
    def _check_suspicious_keywords(self, url):
        """Check for suspicious keywords in the URL."""
        suspicious_words = [
            'secure', 'account', 'webscr', 'login', 'ebayisapi', 'sign-in', 'banking',
            'confirm', 'secure', 'paypal', 'password', 'verification', 'signin'
        ]
        
        url_lower = url.lower()
        for word in suspicious_words:
            if word in url_lower:
                return 1
        return 0
    
    def _extract_domain_features(self, url):
        """Extract features related to the domain."""
        features = {}
        domain = urlparse(url).netloc
        
        # Default values if domain info can't be retrieved
        features['domain_age_days'] = -1
        features['domain_expiry_days'] = -1
        features['domain_registrar'] = "Unknown"
        features['domain_registration_length_days'] = -1
        features['domain_has_ssl'] = 0
        features['domain_ssl_valid'] = 0
        features['domain_ssl_days_to_expiry'] = -1
        
        try:
            # WHOIS information
            domain_info = whois.whois(domain)
            
            # Domain creation date
            if domain_info.creation_date:
                creation_date = domain_info.creation_date
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]
                features['domain_age_days'] = (datetime.now() - creation_date).days
            
            # Domain expiry
            if domain_info.expiration_date:
                expiry_date = domain_info.expiration_date
                if isinstance(expiry_date, list):
                    expiry_date = expiry_date[0]
                features['domain_expiry_days'] = (expiry_date - datetime.now()).days
            
            # Registration length
            if features['domain_age_days'] > 0 and features['domain_expiry_days'] > 0:
                features['domain_registration_length_days'] = features['domain_expiry_days'] + features['domain_age_days']
            
            # Registrar
            if domain_info.registrar:
                features['domain_registrar'] = 1  # We have registrar info
            else:
                features['domain_registrar'] = 0  # No registrar info
                
        except Exception as e:
            # If WHOIS fails, leave the default values
            pass
        
        # SSL Certificate Information
        try:
            hostname = domain
            context = ssl.create_default_context()
            with socket.create_connection((hostname, 443), timeout=3) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    ssl_info = ssock.getpeercert()
                    
                    # Certificate validation
                    features['domain_has_ssl'] = 1
                    features['domain_ssl_valid'] = 1
                    
                    # Days until expiry
                    expire_date = datetime.strptime(ssl_info['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    features['domain_ssl_days_to_expiry'] = (expire_date - datetime.now()).days
        except:
            # If SSL check fails, leave the default values
            pass
        
        return features
    
    def _extract_content_features(self, html_content, url):
        """Extract features from the HTML content of the website."""
        features = {}
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Form-related features
            forms = soup.find_all('form')
            features['form_count'] = len(forms)
            
            features['has_password_field'] = 0
            features['has_hidden_field'] = 0
            features['has_action_external_domain'] = 0
            
            for form in forms:
                if form.find('input', {'type': 'password'}):
                    features['has_password_field'] = 1
                
                if form.find('input', {'type': 'hidden'}):
                    features['has_hidden_field'] = 1
                
                # Check if form action points to another domain
                action = form.get('action', '')
                if action and action.startswith('http'):
                    form_domain = urlparse(action).netloc
                    page_domain = urlparse(url).netloc
                    if form_domain != page_domain:
                        features['has_action_external_domain'] = 1
            
            # Link analysis
            links = soup.find_all('a')
            features['link_count'] = len(links)
            
            external_links = 0
            domain = urlparse(url).netloc
            
            for link in links:
                href = link.get('href', '')
                if href.startswith('http'):
                    link_domain = urlparse(href).netloc
                    if link_domain != domain:
                        external_links += 1
            
            features['external_link_ratio'] = external_links / max(len(links), 1)
            
            # Image analysis
            images = soup.find_all('img')
            features['image_count'] = len(images)
            
            external_images = 0
            for img in images:
                src = img.get('src', '')
                if src.startswith('http'):
                    img_domain = urlparse(src).netloc
                    if img_domain != domain:
                        external_images += 1
            
            features['external_image_ratio'] = external_images / max(len(images), 1)
            
            # Script analysis
            scripts = soup.find_all('script')
            features['script_count'] = len(scripts)
            
            # Page title length
            title = soup.find('title')
            features['title_length'] = len(title.text) if title else 0
            
            # Meta information
            meta_tags = soup.find_all('meta')
            features['meta_tag_count'] = len(meta_tags)
            
            # IFrame presence
            iframes = soup.find_all('iframe')
            features['iframe_count'] = len(iframes)
            
            # Redirect analysis
            features['has_meta_refresh'] = 1 if soup.find('meta', {'http-equiv': re.compile(r'refresh', re.I)}) else 0
            
            # Text statistics
            body_text = soup.get_text()
            features['text_length'] = len(body_text)
            
        except Exception as e:
            # If HTML parsing fails, use default values
            features['form_count'] = 0
            features['has_password_field'] = 0
            features['has_hidden_field'] = 0
            features['has_action_external_domain'] = 0
            features['link_count'] = 0
            features['external_link_ratio'] = 0
            features['image_count'] = 0
            features['external_image_ratio'] = 0
            features['script_count'] = 0
            features['title_length'] = 0
            features['meta_tag_count'] = 0
            features['iframe_count'] = 0
            features['has_meta_refresh'] = 0
            features['text_length'] = 0
            
        return features
    
    def preprocess_data(self, df, extract_features=True):
        """
        Process the dataset by extracting features from URLs.
        
        Args:
            df (pd.DataFrame): DataFrame with 'url' and 'is_phishing' columns
            extract_features (bool): Whether to extract features or use existing ones
            
        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        if extract_features:
            print("Extracting features from URLs...")
            features_list = []
            
            for idx, row in df.iterrows():
                url = row['url']
                try:
                    url_features = self.extract_features(url)
                    url_features['is_phishing'] = row['is_phishing']
                    features_list.append(url_features)
                    
                    if (idx + 1) % 100 == 0:
                        print(f"Processed {idx + 1} URLs")
                except Exception as e:
                    print(f"Error processing URL {url}: {str(e)}")
            
            features_df = pd.DataFrame(features_list)
            return features_df
        else:
            # Assuming features are already in the dataframe
            return df
    
    def train_model(self, features_df, model_type='random_forest', test_size=0.2, random_state=42):
        """
        Train a machine learning model on the feature dataset.
        
        Args:
            features_df (pd.DataFrame): DataFrame with extracted features and labels
            model_type (str): Type of model to train ('random_forest' or 'gradient_boosting')
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing model, evaluation metrics, and other results
        """
        print(f"Training {model_type.replace('_', ' ').title()} model...")
        
        # Prepare data
        X = features_df.drop('is_phishing', axis=1)
        y = features_df['is_phishing']
        
        # Handle non-numeric columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        X = X.drop(categorical_cols, axis=1)
        
        # Store feature names
        self.features = X.columns.tolist()
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        
        # Feature importance for explainability
        if model_type == 'random_forest':
            importances = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': self.features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 important features:")
            print(feature_importance.head(10))
        
        # Return results
        results = {
            'model': self.model,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            },
            'confusion_matrix': cm,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return results
    
    def predict(self, url, html_content=None, explain=False):
        """
        Predict whether a URL is a phishing site.
        
        Args:
            url (str): URL to check
            html_content (str, optional): HTML content if already available
            explain (bool): Whether to provide explanation for the prediction
            
        Returns:
            dict: Prediction results and optionally explanation
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        # Extract features
        features = self.extract_features(url, html_content)
        
        # Convert to DataFrame and handle categorical features
        features_df = pd.DataFrame([features])
        features_df = features_df.reindex(columns=self.features, fill_value=0)
        
        # Scale features
        X = self.scaler.transform(features_df)
        
        # Make prediction
        probability = self.model.predict_proba(X)[0, 1]
        prediction = 1 if probability >= 0.5 else 0
        
        result = {
            'url': url,
            'is_phishing': bool(prediction),
            'probability': float(probability),
            'confidence': float(max(probability, 1 - probability))
        }
        
        # Generate explanation if requested
        if explain and hasattr(self.model, 'feature_importances_'):
            # Simple feature importance based explanation
            importances = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': self.features,
                'Importance': importances,
                'Value': features_df.iloc[0].values
            }).sort_values('Importance', ascending=False)
            
            result['explanation'] = feature_importance.head(10).to_dict('records')
        
        return result
    
    def explain_prediction(self, url, html_content=None):
        """
        Provide a detailed explanation for the prediction using SHAP values.
        
        Args:
            url (str): URL to check
            html_content (str, optional): HTML content if already available
            
        Returns:
            dict: Detailed explanation of the prediction
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        # Extract features
        features = self.extract_features(url, html_content)
        
        # Convert to DataFrame and handle categorical features
        features_df = pd.DataFrame([features])
        features_df = features_df.reindex(columns=self.features, fill_value=0)
        
        # Scale features
        X = self.scaler.transform(features_df)
        
        # Make prediction
        probability = self.model.predict_proba(X)[0, 1]
        prediction = 1 if probability >= 0.5 else 0
        
        # Prepare SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        # Get feature contributions
        if isinstance(shap_values, list):
            # For multi-class models
            feature_contributions = pd.DataFrame({
                'Feature': self.features,
                'Contribution': shap_values[1][0],
                'Value': features_df.iloc[0].values
            })
        else:
            # For binary models
            feature_contributions = pd.DataFrame({
                'Feature': self.features,
                'Contribution': shap_values[0],
                'Value': features_df.iloc[0].values
            })
        
        # Sort by absolute contribution
        feature_contributions['AbsContribution'] = feature_contributions['Contribution'].abs()
        feature_contributions = feature_contributions.sort_values('AbsContribution', ascending=False)
        
        # Get top contributing features
        top_features = feature_contributions.head(10).to_dict('records')
        
        result = {
            'url': url,
            'is_phishing': bool(prediction),
            'probability': float(probability),
            'confidence': float(max(probability, 1 - probability)),
            'top_contributing_features': top_features
        }
        
        return result
    
    def visualize_results(self, results):
        """
        Visualize model evaluation results.
        
        Args:
            results (dict): Results dictionary from train_model method
            
        Returns:
            None (displays plots)
        """
        # Confusion Matrix
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        sns.heatmap(
            results['confusion_matrix'], 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # ROC Curve
        plt.subplot(2, 2, 2)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
        plt.plot(fpr, tpr, lw=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (AUC = {results["metrics"]["roc_auc"]:.3f})')
        
        # Feature Importance (if RandomForest)
        if hasattr(self.model, 'feature_importances_'):
            plt.subplot(2, 2, 3)
            importances = pd.DataFrame({
                'Feature': self.features,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            sns.barplot(x='Importance', y='Feature', data=importances)
            plt.title('Top 10 Feature Importances')
            plt.tight_layout()
        
        # Precision-Recall curve
        plt.subplot(2, 2, 4)
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(results['y_test'], results['y_pred_proba'])
        plt.plot(recall, precision, lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        
        plt.tight_layout()
        plt.show()
    
    def generate_shap_plots(self, X_test, n_samples=100):
        """
        Generate SHAP explainability plots for the model.
        
        Args:
            X_test (np.ndarray): Test feature matrix
            n_samples (int): Number of samples to use for SHAP plots
            
        Returns:
            None (displays plots)
        """
        # Sample a subset of the test data for faster computation
        if X_test.shape[0] > n_samples:
            idx = np.random.choice(X_test.shape[0], size=n_samples, replace=False)
            X_sample = X_test[idx]
        else:
            X_sample = X_test
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list):
            # For multi-class models (use class 1 - phishing)
            shap.summary_plot(shap_values[1], X_sample, feature_names=self.features, show=False)
        else:
            # For binary models
            shap.summary_plot(shap_values, X_sample, feature_names=self.features, show=False)
        
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.show()
        
        # Dependence plots for top features
        if hasattr(self.model, 'feature_importances_'):
            top_features_idx = np.argsort(-self.model.feature_importances_)[:3]
            
            plt.figure(figsize=(15, 5))
            for i, idx in enumerate(top_features_idx):
                plt.subplot(1, 3, i+1)
                if isinstance(shap_values, list):
                    shap.dependence_plot(
                        idx, shap_values[1], X_sample, 
                        feature_names=self.features, show=False
                    )
                else:
                    shap.dependence_plot(
                        idx, shap_values, X_sample, 
                        feature_names=self.features, show=False
                    )
                plt.title(f'Dependence: {self.features[idx]}')
            
            plt.tight_layout()
            plt.show()

def simple_demo():
    """Run a simple demonstration of the phishing detector."""
    print("AI-Powered Phishing Website Detection Demo")
    print("=========================================")
    
    # Sample URLs (for demonstration)
    sample_urls = [
         # Legitimate URLs
        {"url": "https://www.google.com", "is_phishing": 0},
        {"url": "https://facebook.com", "is_phishing": 0},
        {"url": "https://github.com", "is_phishing": 0},
        {"url": "https://www.amazon.com", "is_phishing": 0},
        {"url": "https://www.microsoft.com", "is_phishing": 0},
        {"url": "https://www.apple.com", "is_phishing": 0},
        {"url": "https://www.reddit.com", "is_phishing": 0},
        {"url": "https://www.wikipedia.org", "is_phishing": 0},
        {"url": "https://www.youtube.com", "is_phishing": 0},
        {"url": "https://www.netflix.com", "is_phishing": 0},
        
        # Phishing URLs
        {"url": "http://g00gle.com-secure.info", "is_phishing": 1},
        {"url": "http://paypal-account.secure-billing.com", "is_phishing": 1},
        {"url": "http://online-banking.login.verify.amaz0n.co.uk.info", "is_phishing": 1},
        {"url": "http://facebook-security-login.com", "is_phishing": 1},
        {"url": "http://verification-account.microsoftonline.securelogin.net", "is_phishing": 1},
        {"url": "http://appleid-verify-account-service.com", "is_phishing": 1},
        {"url": "http://secure.bankofamerica.com.logon.verify.accountid.info", "is_phishing": 1},
        {"url": "http://chasebank-verify-account.securedomain.org", "is_phishing": 1},
        {"url": "http://netflix-account-verification.loginportal.site", "is_phishing": 1},
        {"url": "http://instagram-verify.account-security.io", "is_phishing": 1},
        
        # More subtle examples
        {"url": "https://accounts-google.com/login", "is_phishing": 1},
        {"url": "https://www.paypa1.com/signin", "is_phishing": 1},
        {"url": "https://www.dropbox.security-check.com", "is_phishing": 1},
        {"url": "https://mail.google.com.verify-account.ru", "is_phishing": 1},
        {"url": "https://www.instagram-login.help", "is_phishing": 1},
        
        # Legitimate but with suspicious patterns
        {"url": "https://secure.bankofamerica.com", "is_phishing": 0},
        {"url": "https://accounts.google.com/signin", "is_phishing": 0},
        {"url": "https://login.microsoftonline.com", "is_phishing": 0},
        {"url": "https://www.paypal.com/signin", "is_phishing": 0},
        {"url": "https://appleid.apple.com/account", "is_phishing": 0}
    ]
    
    # Create demo dataset
    demo_df = pd.DataFrame(sample_urls)
    
    # Initialize detector
    detector = PhishingDetector()
    
    # Extract features and train model
    features_df = detector.preprocess_data(demo_df)
    results = detector.train_model(features_df, model_type='random_forest')
    
    # Visualize results
    detector.visualize_results(results)
    
    # Test on a new URL
    test_url = "https://www.paypal-secure-login.net.co"
    prediction = detector.predict(test_url, explain=True)
    
    print("\nPrediction for URL:", test_url)
    print(f"Is phishing: {prediction['is_phishing']}")
    print(f"Probability: {prediction['probability']:.4f}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    
    if 'explanation' in prediction:
        print("\nTop features contributing to this decision:")
        for i, feature in enumerate(prediction['explanation'][:5]):
            print(f"{i+1}. {feature['Feature']}: {feature['Importance']:.4f} (Value: {feature['Value']:.4f})")
    
    # Save the model
    detector.save_model("phishing_detector_model.joblib")
    
    print("\nDemo completed successfully.")

if __name__ == "__main__":
    simple_demo()