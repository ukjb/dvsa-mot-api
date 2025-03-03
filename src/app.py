# Required imports
import os
from flask import Flask, request, jsonify
import requests
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    AUTH_URL = os.getenv('AUTH_URL', 'default_auth_url')
    API_BASE_URL = "https://history.mot.api.gov.uk"
    CLIENT_ID = os.getenv('CLIENT_ID')
    CLIENT_SECRET = os.getenv('CLIENT_SECRET')
    SCOPE = "https://tapi.dvsa.gov.uk/.default"
    API_KEY = os.getenv('API_KEY')
    HEADERS = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

class RarityScoreCalculator:
    def __init__(self, csv_path='cleaned_vehicle_statistics.csv'):
        """Initialize the rarity score calculator with vehicle statistics data"""
        try:
            absolute_csv_path = Path(__file__).resolve().parent / csv_path
            
            if not os.path.exists(absolute_csv_path):
                raise FileNotFoundError(f"CSV file not found at path: {absolute_csv_path}")
            
            self.df = pd.read_csv(absolute_csv_path, encoding='utf-8')
            logger.info(f"Initial data load - shape: {self.df.shape}")
            
            self._process_data()
            logger.info("Successfully loaded vehicle statistics data")
            
        except Exception as e:
            logger.error(f"Error loading vehicle statistics: {str(e)}")
            raise

    def _process_data(self):
        """Process the raw data to prepare for rarity calculations"""
        try:
            # Get the latest quarter's data
            quarter_columns = [col for col in self.df.columns if 'Q' in col and col[0].isdigit()]
            self.latest_quarter = max(quarter_columns)
            logger.info(f"Using {self.latest_quarter} as the latest quarter")
            
            # Create clean model names
            self.df['clean_model'] = self.df.apply(
                lambda row: row['GenModel'].replace(row['Make'] + ' ', '', 1) if pd.notnull(row['GenModel']) else '',
                axis=1
            )
            
            # Group by Make and clean_model, combining Licensed and SORN for the latest quarter only
            grouped_df = self.df.groupby(['Make', 'clean_model', 'LicenceStatus'])[self.latest_quarter].sum().reset_index()
            
            # Sum Licensed and SORN vehicles
            final_df = grouped_df.groupby(['Make', 'clean_model'])[self.latest_quarter].sum().reset_index()
            
            # Calculate rarity scores
            total_vehicles = final_df[self.latest_quarter].sum()
            final_df['total_vehicles'] = final_df[self.latest_quarter]
            final_df['rarity_score'] = (1 - (final_df[self.latest_quarter] / total_vehicles)) * 100
            
            # Log statistics
            logger.info(f"Total unique makes: {final_df['Make'].nunique()}")
            logger.info(f"Total vehicles in latest quarter: {total_vehicles:,}")
            logger.info(f"Average vehicles per model: {final_df['total_vehicles'].mean():.0f}")
            
            self.df = final_df
            
        except Exception as e:
            logger.error(f"Error in _process_data: {str(e)}")
            raise

    def get_rarity_score(self, make, model):
        """Get the rarity score for a specific make and model"""
        try:
            logger.info(f"Searching for Make: {make}, Model: {model}")
            
            # Convert to uppercase for case-insensitive matching
            make = make.upper()
            model = model.upper()
            
            # Try exact match
            mask = (self.df['Make'].str.upper() == make) & \
                  (self.df['clean_model'].str.upper() == model)
            
            matches = self.df[mask]
            
            if len(matches) > 0:
                vehicle_data = matches.iloc[0]
                
                # Calculate percentile (lower score = more common)
                percentile = (self.df['rarity_score'] >= vehicle_data['rarity_score']).mean() * 100
                
                result = {
                    'score': float(vehicle_data['rarity_score']),
                    'percentile': round(float(percentile), 2),
                    'total_count': int(vehicle_data['total_vehicles']),
                    'quarter': self.latest_quarter
                }
                
                logger.info(f"Match found for {make} {model}:")
                logger.info(f"Count in {self.latest_quarter}: {result['total_count']:,}")
                logger.info(f"Rarity score: {result['score']:.2f}")
                logger.info(f"Rarer than {result['percentile']:.2f}% of vehicles")
                
                return result
            
            logger.info(f"No match found for Make: {make}, Model: {model}")
            return None
            
        except Exception as e:
            logger.error(f"Error calculating rarity score: {str(e)}")
            return None
            
# Initialize rarity calculator
try:
    rarity_calculator = RarityScoreCalculator()
except Exception as e:
    logger.error(f"Failed to initialize rarity calculator: {str(e)}")
    rarity_calculator = None

def get_oauth_token():
    """Get OAuth token from DVSA authentication service"""
    try:
        payload = {
            'grant_type': 'client_credentials',
            'client_id': Config.CLIENT_ID,
            'client_secret': Config.CLIENT_SECRET,
            'scope': Config.SCOPE
        }
        
        response = requests.post(
            Config.AUTH_URL, 
            data=payload, 
            headers=Config.HEADERS,
            timeout=10
        )
        response.raise_for_status()
        
        token = response.json().get('access_token')
        if not token:
            logger.error("No access token in response")
            return None
            
        return token
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get OAuth token: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/mot/<vehicle_registration>', methods=['GET'])
def get_mot_data(vehicle_registration):
    """Get MOT data with rarity score for a specific vehicle registration"""
    logger.info(f"Receiving request for vehicle: {vehicle_registration}")
    
    if not vehicle_registration or len(vehicle_registration) < 2:
        return jsonify({"error": "Invalid vehicle registration"}), 400
    
    token = get_oauth_token()
    if not token:
        return jsonify({"error": "Failed to authenticate with DVSA service"}), 401
    
    try:
        url = f"{Config.API_BASE_URL}/v1/trade/vehicles/registration/{vehicle_registration}"
        auth_headers = {
            'Authorization': f'Bearer {token}',
            'x-api-key': Config.API_KEY,
            'Content-Type': 'application/json'
        }
        
        response = requests.get(url, headers=auth_headers, timeout=10)
        response.raise_for_status()
        
        vehicle_data = response.json()
        
        if isinstance(vehicle_data, list) and len(vehicle_data) > 0:
            vehicle = vehicle_data[0]
            make = vehicle.get('make')
            model = vehicle.get('model')
            
            # Enhanced debug logging
            logger.info(f"Vehicle data received - Make: {make}, Model: {model}")
            logger.info(f"Rarity calculator initialized: {rarity_calculator is not None}")
            
            if make and model and rarity_calculator:
                # Log the DataFrame state
                if hasattr(rarity_calculator, 'df'):
                    logger.info(f"DataFrame shape: {rarity_calculator.df.shape}")
                    logger.info(f"DataFrame columns: {rarity_calculator.df.columns.tolist()}")
                    logger.info(f"Searching for Make: {make.upper()} in available makes: {rarity_calculator.df['Make'].str.upper().unique()[:5]}")
                    
                    # Log matches in DataFrame
                    make_matches = rarity_calculator.df[rarity_calculator.df['Make'].str.upper() == make.upper()]
                    logger.info(f"Found {len(make_matches)} matches for make {make}")
                    if not make_matches.empty:
                        logger.info(f"Available models for {make}: {make_matches['Model'].unique()[:5]}")
                
                rarity_info = rarity_calculator.get_rarity_score(make, model)
                logger.info(f"Rarity info result: {rarity_info}")
                
                if rarity_info:
                    vehicle_data[0]['rarity_info'] = rarity_info
                    logger.info("Successfully added rarity info to response")
                else:
                    logger.info(f"No rarity info found for Make: {make}, Model: {model}")
                    vehicle_data[0]['rarity_info'] = {
                        'score': None,
                        'percentile': None,
                        'total_count': None,
                        'note': f'Vehicle not found in statistics database (Make: {make}, Model: {model})'
                    }
            else:
                logger.warning("Missing required data for rarity calculation:")
                logger.warning(f"Make present: {bool(make)}")
                logger.warning(f"Model present: {bool(model)}")
                logger.warning(f"Rarity calculator initialized: {bool(rarity_calculator)}")
        
        logger.info(f"Successfully retrieved MOT data for {vehicle_registration}")
        return jsonify(vehicle_data)
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching MOT data: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500
   
	
@app.route('/rarity/<make>/<model>', methods=['GET'])
def get_rarity_score(make, model):
    """Get rarity score for a specific make and model"""
    try:
        if not rarity_calculator:
            return jsonify({"error": "Rarity calculator not initialized"}), 500
            
        rarity_info = rarity_calculator.get_rarity_score(make, model)
        
        if rarity_info:
            return jsonify({
                'make': make,
                'model': model,
                'rarity_info': rarity_info
            })
        else:
            return jsonify({
                'error': 'Vehicle not found in statistics database'
            }), 404
            
    except Exception as e:
        logger.error(f"Error getting rarity score: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Check for required environment variables
    required_vars = ['AUTH_URL', 'CLIENT_ID', 'CLIENT_SECRET', 'API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
