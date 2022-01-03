#!/usr/bin/env python3

from flask import Flask
from flask import request
from classes.api.image_gen.ImageGen import ImageGenerator
import datetime
from tensorflow import keras

app = Flask(__name__)

# Server Config Constants (used for app.run(...))
HOST = 'localhost'
PORT = 8080

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
                'value': 'Server listening on port 3030 !',
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
        model = keras.models.load_model('trained_models/generator')
        image_generator = ImageGenerator(model)

        # Generate image
        image = image_generator.generate_image(target)

        # TODO: Encode image data into base64

        # Build response
        response = {
            'status': 200,
            'timestamp': datetime.datetime.now(),
            'route': 'api/v1/',
            'data': {
                'image_data': {
                    'value': image,
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
      app.run(debug=True)