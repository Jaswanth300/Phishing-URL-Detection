import re
from urllib.parse import urlparse

def extract_features(url):
    features = {}

    parsed = urlparse(url)
    domain = parsed.netloc

    # Basic length feature
    features['url_length'] = len(url)

    # Count dots
    features['num_dots'] = url.count('.')

    # Check for @ symbol
    features['has_at'] = 1 if '@' in url else 0

    # Check for hyphen in domain
    features['has_hyphen'] = 1 if '-' in domain else 0

    # HTTPS check
    features['https'] = 1 if url.startswith("https") else 0

    # Suspicious keywords
    suspicious_words = ["login", "verify", "secure", "account", "update", "bank"]
    features['suspicious_words'] = sum(word in url.lower() for word in suspicious_words)

    # IP address detection
    ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'
    features['has_ip'] = 1 if re.search(ip_pattern, domain) else 0

    return features
