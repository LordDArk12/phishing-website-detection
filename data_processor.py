"""
Data Collection and Preprocessing Module for Phishing Website Detection
This module handles data collection, preparation, and feature extraction for the
phishing detection system.
"""

import os
import pandas as pd
import numpy as np
import requests
from urllib.parse import urlparse
import re
import time
import random
from bs4 import BeautifulSoup
from tqdm import tqdm
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

class DataProcessor:
    """
    Handles data collection, preprocessing, and feature extraction for phishing
    website detection.
    """
    
    def __init__(self):
        """Initialize the DataProcessor."""
        # User-agent for requests
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        ]
    
    def get_random_user_agent(self):
        """Return a random user agent from the list."""
        return random.choice(self.user_agents)
    
    def collect_phishtank_data(self, api_key=None, max_entries=1000):
        """
        Collect phishing URLs from PhishTank API.
        
        Args:
            api_key (str, optional): PhishTank API key
            max_entries (int): Maximum number of entries to collect
            
        Returns:
            pd.DataFrame: DataFrame with collected URLs and labels
        """
        print("Collecting data from PhishTank...")
        
        url = "http://data.phishtank.com/data/"
        if api_key:
            url += f"{api_key}/"
        url += "online-valid.csv"
        
        try:
            df = pd.read_csv(url)
            print(f"Downloaded {len(df)} entries from PhishTank")
            
            # Extract relevant columns
            phishing_data = pd.DataFrame({
                'url': df['url'].values,
                'is_phishing': 1  # All URLs from PhishTank are phishing
            })
            
            # Limit entries if needed
            if len(phishing_data) > max_entries:
                phishing_data = phishing_data.sample(max_entries, random_state=42)
            
            return phishing_data
            
        except Exception as e:
            print(f"Error collecting data from PhishTank: {str(e)}")
            return pd.DataFrame(columns=['url', 'is_phishing'])
    
    def collect_alexa_top_sites(self, max_entries=1000):
        """
        Collect top legitimate websites from Alexa Top Sites.
        
        Args:
            max_entries (int): Maximum number of entries to collect
            
        Returns:
            pd.DataFrame: DataFrame with collected URLs and labels
        """
        print("Collecting data from Alexa Top Sites...")
        
        # Alternative to Alexa Top Sites (since Alexa has been discontinued)
        top_sites_url = "https://s3.amazonaws.com/alexa-static/top-1m.csv.zip"
        
        try:
            # Download and read the CSV file
            df = pd.read_csv(top_sites_url, compression='zip', header=None, nrows=max_entries)
            df.columns = ['rank', 'domain']
            
            # Convert domains to URLs
            legitimate_data = pd.DataFrame({
                'url': ['https://' + domain for domain in df['domain']],
                'is_phishing': 0  # All URLs from Alexa Top Sites are legitimate
            })
            
            print(f"Collected {len(legitimate_data)} legitimate websites")
            return legitimate_data
            
        except Exception as e:
            print(f"Error collecting data from Alexa Top Sites: {str(e)}")
            
            # Fallback to a predefined list of known legitimate websites
            legitimate_sites = [
                "https://www.google.com",
                "https://www.youtube.com",
                "https://www.facebook.com",
                "https://www.amazon.com",
                "https://www.wikipedia.org",
                "https://www.twitter.com",
                "https://www.instagram.com",
                "https://www.linkedin.com",
                "https://www.microsoft.com",
                "https://www.apple.com",
                # Add more known legitimate sites here
            ]
            
            # Create DataFrame with legitimate URLs
            legitimate_data = pd.DataFrame({
                'url': legitimate_sites,
                'is_phishing': 0
            })
            
            print(f"Used fallback list with {len(legitimate_data)} legitimate websites")
            return legitimate_data
    
    def collect_openphish_data(self, max_entries=1000):
        """
        Collect phishing URLs from OpenPhish feed.
        
        Args:
            max_entries (int): Maximum number of entries to collect
            
        Returns:
            pd.DataFrame: DataFrame with collected URLs and labels
        """
        print("Collecting data from OpenPhish...")
        
        # Note: OpenPhish may require paid subscription for access
        # Using a placeholder approach here
        try:
            # This would be the URL for the OpenPhish feed
            feed_url = "https://openphish.com/feed.txt"
            
            response = requests.get(feed_url, 
                                   headers={'User-Agent': self.get_random_user_agent()},
                                   timeout=10)
            
            if response.status_code == 200:
                # Extract URLs from the text feed
                phishing_urls = response.text.strip().split('\n')
                
                # Limit entries if needed
                if len(phishing_urls) > max_entries:
                    phishing_urls = phishing_urls[:max_entries]
                
                # Create DataFrame
                phishing_data = pd.DataFrame({
                    'url': phishing_urls,
                    'is_phishing': 1  # All URLs from OpenPhish are phishing
                })
                
                print(f"Collected {len(phishing_data)} phishing URLs from OpenPhish")
                return phishing_data
            else:
                print(f"Failed to access OpenPhish feed: Status code {response.status_code}")
                return pd.DataFrame(columns=['url', 'is_phishing'])
                
        except Exception as e:
            print(f"Error collecting data from OpenPhish: {str(e)}")
            return pd.DataFrame(columns=['url', 'is_phishing'])
    
    def collect_custom_data(self, phishing_path, legitimate_path=None):
        """
        Load custom datasets from local files.
        
        Args:
            phishing_path (str): Path to CSV file with phishing URLs
            legitimate_path (str, optional): Path to CSV file with legitimate URLs
            
        Returns:
            pd.DataFrame: DataFrame with collected URLs and labels
        """
        print("Loading custom datasets...")
        
        data_frames = []
        
        # Load phishing URLs
        if os.path.exists(phishing_path):
            try:
                phishing_df = pd.read_csv(phishing_path)
                
                # Check if the CSV has the expected columns
                if 'url' in phishing_df.columns:
                    phishing_data = pd.DataFrame({
                        'url': phishing_df['url'].values,
                        'is_phishing': 1
                    })
                    data_frames.append(phishing_data)
                    print(f"Loaded {len(phishing_data)} phishing URLs from {phishing_path}")
                else:
                    print(f"Error: {phishing_path} does not contain 'url' column")
            except Exception as e:
                print(f"Error loading phishing data from {phishing_path}: {str(e)}")
        else:
            print(f"Warning: Phishing data file {phishing_path} not found")
        
        # Load legitimate URLs if provided
        if legitimate_path and os.path.exists(legitimate_path):
            try:
                legitimate_df = pd.read_csv(legitimate_path)
                
                # Check if the CSV has the expected columns
                if 'url' in legitimate_df.columns:
                    legitimate_data = pd.DataFrame({
                        'url': legitimate_df['url'].values,
                        'is_phishing': 0
                    })
                    data_frames.append(legitimate_data)
                    print(f"Loaded {len(legitimate_data)} legitimate URLs from {legitimate_path}")
                else:
                    print(f"Error: {legitimate_path} does not contain 'url' column")
            except Exception as e:
                print(f"Error loading legitimate data from {legitimate_path}: {str(e)}")
        elif legitimate_path:
            print(f"Warning: Legitimate data file {legitimate_path} not found")
        
        # Combine datasets if any were loaded
        if data_frames:
            combined_df = pd.concat(data_frames, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame(columns=['url', 'is_phishing'])
    
    def collect_data(self, sources=None, max_entries_per_source=1000):
        """
        Collect data from multiple sources and combine them.
        
        Args:
            sources (list, optional): List of sources to collect from
                                      ('phishtank', 'alexa', 'openphish', 'custom')
            max_entries_per_source (int): Maximum entries to collect from each source
            
        Returns:
            pd.DataFrame: Combined DataFrame with URLs and labels
        """
        if sources is None:
            sources = ['phishtank', 'alexa']
        
        data_frames = []
        
        # Collect data from each specified source
        for source in sources:
            if source == 'phishtank':
                df = self.collect_phishtank_data(max_entries=max_entries_per_source)
                if not df.empty:
                    data_frames.append(df)
            
            elif source == 'alexa':
                df = self.collect_alexa_top_sites(max_entries=max_entries_per_source)
                if not df.empty:
                    data_frames.append(df)
            
            elif source == 'openphish':
                df = self.collect_openphish_data(max_entries=max_entries_per_source)
                if not df.empty:
                    data_frames.append(df)
            
            elif source == 'custom':
                df = self.collect_custom_data(
                    phishing_path='data/custom_phishing.csv',
                    legitimate_path='data/custom_legitimate.csv'
                )
                if not df.empty:
                    data_frames.append(df)
        
        # Combine all datasets
        if data_frames:
            combined_df = pd.concat(data_frames, ignore_index=True)
            print(f"Combined dataset contains {len(combined_df)} URLs")
            
            # Remove duplicates
            combined_df.drop_duplicates(subset=['url'], inplace=True)
            print(f"After removing duplicates: {len(combined_df)} URLs")
            
            return combined_df
        else:
            print("No data was collected from any source")
            return pd.DataFrame(columns=['url', 'is_phishing'])
    
    def fetch_html_content(self, url, timeout=5, max_retries=2):
        """
        Fetch HTML content from a URL with retry mechanism.
        
        Args:
            url (str): URL to fetch
            timeout (int): Request timeout in seconds
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            str: HTML content or empty string if failed
        """
        headers = {'User-Agent': self.get_random_user_agent()}
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(url, headers=headers, timeout=timeout, verify=False)
                if response.status_code == 200:
                    return response.text
                else:
                    print(f"Failed to fetch {url}: Status code {response.status_code}")
            except Exception as e:
                if attempt < max_retries:
                    print(f"Error fetching {url}: {str(e)}. Retrying...")
                    time.sleep(1)  # Wait before retrying
                else:
                    print(f"Failed to fetch {url} after {max_retries} retries: {str(e)}")
        
        return ""
    
    def enrich_dataset_with_html(self, df, html_column='html_content', sample_size=None):
        """
        Fetch HTML content for URLs in the dataset.
        
        Args:
            df (pd.DataFrame): DataFrame with 'url' column
            html_column (str): Name of the column to store HTML content
            sample_size (int, optional): Number of URLs to sample for enrichment
            
        Returns:
            pd.DataFrame: DataFrame with added HTML content column
        """
        # Create a copy to avoid modifying the original
        enriched_df = df.copy()
        
        # Add HTML content column
        if html_column not in enriched_df.columns:
            enriched_df[html_column] = ""
        
        # Sample URLs if specified
        if sample_size and len(enriched_df) > sample_size:
            # Ensure we have a balance of phishing and legitimate URLs
            phishing_df = enriched_df[enriched_df['is_phishing'] == 1]
            legitimate_df = enriched_df[enriched_df['is_phishing'] == 0]
            
            phishing_sample = phishing_df.sample(
                min(sample_size // 2, len(phishing_df)), 
                random_state=42
            )
            legitimate_sample = legitimate_df.sample(
                min(sample_size // 2, len(legitimate_df)), 
                random_state=42
            )
            
            enriched_df = pd.concat([phishing_sample, legitimate_sample], ignore_index=True)
        
        print(f"Fetching HTML content for {len(enriched_df)} URLs...")
        
        # Fetch HTML content for each URL
        for idx, row in tqdm(enriched_df.iterrows(), total=len(enriched_df)):
            url = row['url']
            html_content = self.fetch_html_content(url)
            enriched_df.at[idx, html_column] = html_content
            
            # Throttle requests to avoid overloading servers
            time.sleep(0.5)
        
        return enriched_df
    
    def balance_dataset(self, df, method='undersample'):
        """
        Balance the dataset to have equal number of phishing and legitimate URLs.
        
        Args:
            df (pd.DataFrame): DataFrame with 'is_phishing' column
            method (str): Balancing method ('undersample', 'oversample', or 'smote')
            
        Returns:
            pd.DataFrame: Balanced DataFrame
        """
        print(f"Balancing dataset using {method}...")
        
        # Count number of samples in each class
        phishing_count = len(df[df['is_phishing'] == 1])
        legitimate_count = len(df[df['is_phishing'] == 0])
        
        print(f"Original dataset: {phishing_count} phishing, {legitimate_count} legitimate")
        
        # Extract phishing and legitimate examples
        phishing_df = df[df['is_phishing'] == 1]
        legitimate_df = df[df['is_phishing'] == 0]
        
        if method == 'undersample':
            # Undersample the majority class
            if phishing_count > legitimate_count:
                phishing_df = phishing_df.sample(legitimate_count, random_state=42)
            else:
                legitimate_df = legitimate_df.sample(phishing_count, random_state=42)
            
            # Combine balanced datasets
            balanced_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)
            
        elif method == 'oversample':
            # Oversample the minority class
            if phishing_count < legitimate_count:
                phishing_df = resample(
                    phishing_df, 
                    replace=True, 
                    n_samples=legitimate_count, 
                    random_state=42
                )
            else:
                legitimate_df = resample(
                    legitimate_df, 
                    replace=True, 
                    n_samples=phishing_count, 
                    random_state=42
                )
            
            # Combine balanced datasets
            balanced_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)
            
        elif method == 'smote':
            # Apply SMOTE for more sophisticated oversampling
            # Note: This requires removing any non-numeric columns first
            
            # Temporary dataframe for SMOTE
            features_df = df.drop(['url', 'is_phishing'], axis=1, errors='ignore')
            
            # Check if there are features to work with
            if features_df.empty:
                print("Cannot apply SMOTE: No feature columns found")
                return df
            
            # Handle non-numeric columns
            categorical_cols = features_df.select_dtypes(include=['object']).columns
            features_df = features_df.drop(categorical_cols, axis=1)
            
            # Check if there are still features after dropping categorical
            if features_df.empty:
                print("Cannot apply SMOTE: No numeric feature columns found")
                return df
            
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(features_df, df['is_phishing'])
            
            # Reconstruct DataFrame
            balanced_df = pd.DataFrame(X_resampled, columns=features_df.columns)
            balanced_df['is_phishing'] = y_resampled
            
            # Add URL column back (for the samples that were not synthetic)
            url_mapping = dict(zip(df.index, df['url']))
            balanced_df['url'] = balanced_df.index.map(
                lambda x: url_mapping.get(x, f"synthetic_url_{x}")
            )
            
        else:
            print(f"Unknown balancing method: {method}")
            return df
        
        print(f"Balanced dataset: {len(balanced_df[balanced_df['is_phishing'] == 1])} phishing, "
              f"{len(balanced_df[balanced_df['is_phishing'] == 0])} legitimate")
        
        return balanced_df
    
    def generate_synthetic_features(self, df, num_samples=100):
        """
        Generate synthetic features for improved model training.

        Args:
            df (pd.DataFrame): DataFrame with features
            num_samples (int): Number of synthetic samples to generate
            
        Returns:
            pd.DataFrame: DataFrame with added synthetic samples
        """
        print(f"Generating {num_samples} synthetic samples...")
        
        # Check if there are feature columns
        feature_cols = df.columns.difference(['url', 'is_phishing', 'html_content'])
        if len(feature_cols) == 0:
            print("Cannot generate synthetic features: No feature columns found")
            return df
        
        # Separate by class
        phishing_df = df[df['is_phishing'] == 1]
        legitimate_df = df[df['is_phishing'] == 0]
        
        synthetic_samples = []
        
        # Generate synthetic phishing samples
        for _ in range(num_samples // 2):
            # Sample random phishing record
            sample = phishing_df.sample(1).iloc[0]
            
            # Create new synthetic sample
            synthetic_sample = {}
            synthetic_sample['is_phishing'] = 1
            synthetic_sample['url'] = f"synthetic_phishing_{_}"
            
            # Copy and modify features
            for col in feature_cols:
                if col in ['url', 'is_phishing', 'html_content']:
                    continue
                
                # Numeric features: add small random variation
                if pd.api.types.is_numeric_dtype(df[col]):
                    base_value = sample[col]
                    if pd.notna(base_value):
                        # Add random noise (±20%)
                        synthetic_sample[col] = base_value * (1 + 0.2 * (2 * random.random() - 1))
                    else:
                        synthetic_sample[col] = base_value
                else:
                    # Non-numeric: just copy
                    synthetic_sample[col] = sample[col]
            
            synthetic_samples.append(synthetic_sample)
        
        # Generate synthetic legitimate samples
        for _ in range(num_samples - num_samples // 2):
            # Sample random legitimate record
            sample = legitimate_df.sample(1).iloc[0]
            
            # Create new synthetic sample
            synthetic_sample = {}
            synthetic_sample['is_phishing'] = 0
            synthetic_sample['url'] = f"synthetic_legitimate_{_}"
            
            # Copy and modify features
            for col in feature_cols:
                if col in ['url', 'is_phishing', 'html_content']:
                    continue
                
                # Numeric features: add small random variation
                if pd.api.types.is_numeric_dtype(df[col]):
                    base_value = sample[col]
                    if pd.notna(base_value):
                        # Add random noise (±20%)
                        synthetic_sample[col] = base_value * (1 + 0.2 * (2 * random.random() - 1))
                    else:
                        synthetic_sample[col] = base_value
                else:
                    # Non-numeric: just copy
                    synthetic_sample[col] = sample[col]
            
            synthetic_samples.append(synthetic_sample)
        
        # Convert to DataFrame and combine with original
        synthetic_df = pd.DataFrame(synthetic_samples)
        combined_df = pd.concat([df, synthetic_df], ignore_index=True)
        
        print(f"Added {len(synthetic_df)} synthetic samples")
        
        return combined_df
    
    def save_dataset(self, df, filepath, include_html=False):
        """
        Save the dataset to a CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filepath (str): Path to save the CSV file
            include_html (bool): Whether to include HTML content (can make file large)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Create a copy to avoid modifying the original
            save_df = df.copy()
            
            # Remove HTML content if not required
            if not include_html and 'html_content' in save_df.columns:
                save_df.drop('html_content', axis=1, inplace=True)
            
            # Save to CSV
            save_df.to_csv(filepath, index=False)
            print(f"Dataset saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving dataset to {filepath}: {str(e)}")
            return False
    
    def load_dataset(self, filepath):
        """
        Load a dataset from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded DataFrame or None if failed
        """
        try:
            if not os.path.exists(filepath):
                print(f"Dataset file {filepath} not found")
                return None
            
            df = pd.read_csv(filepath)
            print(f"Loaded dataset from {filepath} with {len(df)} records")
            return df
            
        except Exception as e:
            print(f"Error loading dataset from {filepath}: {str(e)}")
            return None

    def generate_test_dataset(self, size=100):
        """
        Generate a small test dataset for development and testing.
        
        Args:
            size (int): Number of records to generate
            
        Returns:
            pd.DataFrame: Generated test dataset
        """
        print(f"Generating test dataset with {size} records...")
        
        test_data = []
        
        # Generate legitimate URLs
        legitimate_domains = [
            "google.com", "facebook.com", "twitter.com", "amazon.com",
            "youtube.com", "instagram.com", "linkedin.com", "microsoft.com",
            "apple.com", "github.com", "wikipedia.org", "reddit.com",
            "netflix.com", "yahoo.com", "ebay.com", "cnn.com"
        ]
        
        for i in range(size // 2):
            domain = random.choice(legitimate_domains)
            path = "" if random.random() < 0.3 else f"/{random.choice(['about', 'contact', 'help', 'login', 'products'])}"
            protocol = "https" if random.random() < 0.9 else "http"
            
            url = f"{protocol}://www.{domain}{path}"
            test_data.append({
                'url': url,
                'is_phishing': 0
            })
        
        # Generate phishing URLs
        for i in range(size - size // 2):
            # Choose a method to generate phishing URL
            method = random.randint(1, 4)
            
            if method == 1:
                # Typo in domain
                domain = random.choice(legitimate_domains)
                # Introduce typo
                typo_domain = list(domain)
                pos = random.randint(0, len(domain) - 2)
                typo_domain[pos] = random.choice(['a', 'e', 'i', 'o', 'u', '0', '1'])
                typo_domain = ''.join(typo_domain)
                
                url = f"http://www.{typo_domain}"
                
            elif method == 2:
                # Suspicious subdomains
                domain = random.choice(legitimate_domains)
                subdomain = random.choice(['secure', 'login', 'account', 'signin', 'banking'])
                tld = random.choice(['.com', '.net', '.org', '.info', '.tk'])
                
                url = f"http://{domain}.{subdomain}{tld}"
                
            elif method == 3:
                # Long domain with keywords
                domain = random.choice([
                    "secure-login-account-verify",
                    "account-verification-secure",
                    "login-secure-validation",
                    "authentication-secure-verify"
                ])
                tld = random.choice(['.com', '.net', '.org', '.info', '.tk'])
                
                url = f"http://{domain}{tld}"
                
            else:
                # Using IP address
                ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
                path = f"/{random.choice(['secure', 'login', 'account', 'signin', 'banking'])}"
                
                url = f"http://{ip}{path}"
            
            test_data.append({
                'url': url,
                'is_phishing': 1
            })
        
        # Convert to DataFrame and shuffle
        test_df = pd.DataFrame(test_data)
        test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return test_df

# Example usage
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Generate test dataset
    test_df = processor.generate_test_dataset(size=100)
    
    # Save test dataset
    processor.save_dataset(test_df, "data/test_dataset.csv")
    
    print("Test dataset generated and saved.")