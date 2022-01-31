#!/usr/bin/env python3

from os import stat
import unittest
from ..ImageGen import ImageGenerator
from tensorflow import keras
from PIL import Image

class ImageGenTestCase(unittest.TestCase):

    def setUp(self):
        self.imagegen = None
        try:
            # Load model from root/trained_models/
            api_object = ImageGenerator(keras.models.load_model('trained_models/generator', compile=False))
            self.imagegen =  api_object
        except OSError:
            self.fail('OSError : ImageGen API object not initialized. Model file not found.')

    def test_init(self):
        """
            Tests the constructor by checking the value of a variable before and after object allocation
        """
        self.assertIsNotNone(self.imagegen)
    
    def test_get_model_summary(self):
        """
            Tests the get_model_summary() function of the ImageGen API based on the returned value
        """
        status = self.imagegen.get_model_summary()
        self.assertTrue(status[0], status[1])
    
    def test_generate_image(self):
        """
            Tests the generate_image() function with all possible args of the ImageGen API 
            by trying to cast the resulting np.array
            to a PIL image and checking variable change
        """
        image_to_test = self.imagegen.generate_image(target='rock')
        pil_image = None
        pil_image = Image.fromarray(image_to_test, 'RGB')
        self.assertIsNotNone(pil_image)
    
if __name__ == '__main__':
    unittest.main(verbosity=2)