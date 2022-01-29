#!/usr/bin/env python3

from os import environ # Get access to environment variables
from flask import Flask
from flask import request
from classes.api.image_gen.ImageGen import ImageGenerator
import datetime
from tensorflow import keras
import numpy as np
import json

app = Flask(__name__)

# Server Config Constants (used for app.run(...))
HOST = '0.0.0.0'

### API V1 Routes ###
@app.route('/api/v1/')
def api_v1_access_test():
    
    # Standard response format in all cases
    response = {
        'status': 200,
        'timestamp': datetime.datetime.now(),
        'route': 'api/v1/',
        'data': {
            'image_data': {},
            'text_data': {
                'value': f'Server listening on port {environ.get("PORT", 5050)} !',
            }
        }
    }

    return response

@app.route('/api/v1/generate')
def api_v1_generate():

    target = request.args.get('target')

    # Sanitise target str
    target = target.lower()
    if target != 'rock' and target != 'paper' and target != 'scissors':
        return {
            'status': 500,
            'timestamp': datetime.datetime.now(),
            'route': 'api/v1/generate',
            'data': {
                'image_data': {},
                'text_data': {
                    'value': 'Target has to be a one of these words : rock, paper, scissors',
                    'target': target
                }
            }
        }
    
    try:
        # Load model
        model = keras.models.load_model('trained_models/generator', compile=False)
        image_generator = ImageGenerator(model)

        # Generate image
        image = image_generator.generate_image(target)

        # TODO: Encode image data into base64
        image = np.array(image)
        image_data_lists = image.tolist()
        json_image = json.dumps(image_data_lists)

        # Build response
        response = {
            'status': 200,
            'timestamp': datetime.datetime.now(),
            'route': 'api/v1/',
            'data': {
                'image_data': {
                    'value': json_image,
                    'target': target
                },
                'text_data': {}
            }
        }

        return response

    except OSError:
            # TODO: Better error messages
            return {
                'status': 500,
                'timestamp': datetime.datetime.now(),
                'route': 'api/v1/generate',
                'data': {
                    'image_data': {},
                    'text_data': {
                        'value': 'Model file not found.'
                    }
                }
            }

### App Run ###
if __name__ == '__main__':
      app.run(debug=False, host=HOST, port=environ.get("PORT", 5050))