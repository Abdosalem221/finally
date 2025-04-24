
import os
import requests
from typing import Dict, Optional

def validate_api_keys() -> Dict[str, bool]:
    """التحقق من صلاحية مفاتيح API"""
    validation_results = {}
    
    # AlphaVantage
    alpha_key = os.getenv('ALPHAVANTAGE_API_KEY')
    if alpha_key:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey={alpha_key}'
        response = requests.get(url)
        validation_results['alphavantage'] = response.status_code == 200
    
    # FCS API
    fcs_key = os.getenv('FCSAPI_API_KEY')
    if fcs_key:
        url = f'https://fcsapi.com/api-v3/forex/latest?symbol=EUR/USD&access_key={fcs_key}'
        response = requests.get(url)
        validation_results['fcsapi'] = response.status_code == 200
    
    # Twelve Data
    twelve_key = os.getenv('TWELVEDATA_API_KEY')
    if twelve_key:
        url = f'https://api.twelvedata.com/time_series?symbol=EUR/USD&interval=1min&apikey={twelve_key}'
        response = requests.get(url)
        validation_results['twelvedata'] = response.status_code == 200
    
    return validation_results

def check_required_keys() -> Optional[str]:
    """التحقق من وجود جميع المفاتيح المطلوبة"""
    required_keys = [
        'ALPHAVANTAGE_API_KEY',
        'FCSAPI_API_KEY', 
        'TWELVEDATA_API_KEY',
        'TELEGRAM_API_TOKEN',
        'SECRET_KEY'
    ]
    
    missing_keys = []
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        return f"Missing required keys: {', '.join(missing_keys)}"
    return None
