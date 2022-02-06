#!/usr/bin/env python3

import sys
import os
# Setting runtime path to server/ to access the classes module
sys.path.insert(1, os.path.join(sys.path[0], 'server'))

# Imports
import os
from os import environ # Get access to environment variables
import base64
from flask import Flask
from flask import request
from classes.api.image_gen.ImageGen import ImageGenerator
from tensorflow import keras
from PIL import Image
from io import BytesIO
from classes.rest.Response import Response

app = Flask(__name__)

# Server Config Constants (used for app.run(...))
HOST = '0.0.0.0'

### API V1 Routes ###
@app.route('/api/v1/')
def api_v1_access_test():
    
    return Response.response(                                                                           \
        status=200,                                                                                     \
        route='api/v1/',                                                                                \
        data= {
            'image_data': {}, 
            'text_data': {'value': f'Server listening on port {environ.get("PORT", 5050)} !'}
        }
    )

@app.route('/api/v1/generate')
def api_v1_generate(target_arg=None):

    # target_arg used by the server test case, to avoid running a server subprocess
    # in GitHub Actions and making HTTP requests
    if target_arg == None:
        target = request.args.get('target')
    else:
        target = target_arg

    # Sanitise target str
    target = target.lower()
    if target != 'rock' and target != 'paper' and target != 'scissors':
        return Response.response(                                                                       \
            status=500,                                                                                 \
            route='api/v1/generate',                                                                    \
            data= {
                'image_data': {}, 
                'text_data': {
                    'value': 'Target has to be a one of these words (case-insensitive) : rock, paper, scissors',
                    'target': target
                }
            }
        )      
    
    try:
        # Load model from root/trained_models/
        model = keras.models.load_model('trained_models/generator', compile=False)
        image_generator = ImageGenerator(model)

        # Generate image
        image = image_generator.generate_image(target)

        # Encode image data into base64 using PIL and a byte buffer
        pil_image = Image.fromarray(image, 'RGB')
        buffered = BytesIO()
        pil_image.save(buffered, format='JPEG')
        base64_string_image = base64.b64encode(buffered.getvalue()).decode('ascii')

        return Response.response(                                                                       \
            status=200,                                                                                 \
            route='api/v1/generate',                                                                    \
            data= {
                'image_data': {
                    'value': base64_string_image,
                    'target': target
                }, 
                'text_data': {}
            }
        ) 

    except OSError:
            # TODO: Better error messages
            return Response.response(                                                                   \
                status=500,                                                                             \
                route='api/v1/generate',                                                                \
                data= {
                    'image_data': {}, 
                    'text_data': {
                        'value': 'Model file not found',
                        'location': os.getcwd()
                    }
                }
            ) 

### App Run ###
if __name__ == '__main__':
      app.run(debug=False, host=HOST, port=environ.get("PORT", 5050))