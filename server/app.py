#!/usr/bin/env python3

# Imports
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
    
    return Response.response(                                                                   \
        status=200,                                                                             \
        route='api/v1/',                                                                        \
        data= {
            'image_data': {}, 
            'text_data': {'value': f'Server listening on port {environ.get("PORT", 5050)} !'}
        }
    )

@app.route('/api/v1/generate')
def api_v1_generate():

    target = request.args.get('target')

    # Sanitise target str
    target = target.lower()
    if target != 'rock' and target != 'paper' and target != 'scissors':
        return Response.response(                                                                   \
            status=500,                                                                             \
            route='api/v1/generate',                                                                \
            data= {
                'image_data': {}, 
                'text_data': {
                    'value': 'Target has to be a one of these words : rock, paper, scissors',
                    'target': target
                }
            }
        )      
    
    try:
        # Load model
        model = keras.models.load_model('trained_models/generator', compile=False)
        image_generator = ImageGenerator(model)

        # Generate image
        image = image_generator.generate_image(target)

        # Encode image data into base64 using PIL and a byte buffer
        pil_image = Image.fromarray(image, 'RGB')
        buffered = BytesIO()
        pil_image.save(buffered, format='JPEG')
        base64_string_image = base64.b64encode(buffered.getvalue()).decode('ascii')

        return Response.response(                                                                   \
            status=200,                                                                             \
            route='api/v1/generate',                                                                \
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
                        'value': 'Model file not found.'
                    }
                }
            ) 

### App Run ###
if __name__ == '__main__':
      app.run(debug=False, host=HOST, port=environ.get("PORT", 5050))