import requests
import argparse
import json
import time
import logging
from typing import Dict, Any
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CerebriumTester:
    """Class for testing deployed model on Cerebrium."""
    
    def __init__(self, api_key: str, api_url: str):
        """
        Initialize Cerebrium tester.
        
        Args:
            api_key (str): Cerebrium API key
            api_url (str): Cerebrium API endpoint URL
        """
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {api_key}"
        }
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Make prediction request to deployed model.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            Dict[str, Any]: Prediction response
        """
        try:
            # Prepare the file for upload
            files = {
                'file': ('image.jpg', open(image_path, 'rb'), 'image/jpeg')
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                files=files
            )
            
            # Log the response content for debugging
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response content: {response.text}")
            
            response.raise_for_status()
            return response.json()
                
        except Exception as e:
            logger.error(f"Error making prediction request: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise
    
    def run_tests(self, test_images: Dict[str, int]) -> None:
        """
        Run test cases on deployed model.
        
        Args:
            test_images (Dict[str, int]): Dictionary mapping image paths to expected class IDs
        """
        results = []
        
        for image_path, expected_class in test_images.items():
            try:
                start_time = time.time()
                response = self.predict(image_path)
                end_time = time.time()
                
                predicted_class = response.get('class_id')
                confidence = response.get('confidence', 0)
                inference_time = response.get('inference_time', 0)
                
                response_time = end_time - start_time
                
                is_correct = predicted_class == expected_class
                
                result = {
                    'image': image_path,
                    'expected_class': expected_class,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'inference_time': inference_time,
                    'response_time': response_time,
                    'is_correct': is_correct
                }
                
                results.append(result)
                
                logger.info(f"Test result for {image_path}:")
                logger.info(f"  Expected class: {expected_class}")
                logger.info(f"  Predicted class: {predicted_class}")
                logger.info(f"  Confidence: {confidence:.4f}")
                logger.info(f"  Inference time: {inference_time:.4f}s")
                logger.info(f"  Response time: {response_time:.4f}s")
                logger.info(f"  Correct prediction: {is_correct}")
                
            except Exception as e:
                logger.error(f"Error testing {image_path}: {str(e)}")
                results.append({
                    'image': image_path,
                    'error': str(e)
                })
        
        successful_tests = sum(1 for r in results if r.get('is_correct', False))
        total_tests = len(test_images)
        
        logger.info("\nTest Summary:")
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Successful predictions: {successful_tests}")
        logger.info(f"Success rate: {(successful_tests/total_tests)*100:.2f}%")
        
        avg_response_time = sum(r.get('response_time', 0) for r in results) / len(results)
        logger.info(f"Average response time: {avg_response_time:.4f}s")
        
        if avg_response_time > 3:
            logger.warning("Average response time exceeds 3 seconds!")

def main():
    parser = argparse.ArgumentParser(description='Test deployed model on Cerebrium')
    parser.add_argument('--image_path', type=str, help='Path to single image for testing')
    parser.add_argument('--run_tests', action='store_true', help='Run all test cases')
    
    args = parser.parse_args()
    
    # Get API credentials from environment
    api_key = os.getenv('CEREBRIUM_API_KEY')
    api_url = os.getenv('CEREBRIUM_API_URL')
    
    if not api_key or not api_url:
        raise ValueError("CEREBRIUM_API_KEY and CEREBRIUM_API_URL must be set in environment")
    
    tester = CerebriumTester(api_key, api_url)
    
    if args.run_tests:
        test_images = {
            "assets/n01440764_tench.jpeg": 0,
            "assets/n01667114_mud_turtle.JPEG": 35
        }
        tester.run_tests(test_images)
    elif args.image_path:
        response = tester.predict(args.image_path)
        print(json.dumps(response, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 