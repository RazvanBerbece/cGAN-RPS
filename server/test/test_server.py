#!/usr/bin/env python3

import base64
import unittest
from enum import Enum
from PIL import Image
from server.app import api_v1_access_test, api_v1_generate
from io import BytesIO

class Targets(Enum):
    """
        Target strings for requests annd their respective expected status code
    """
    ROCK_TARGET = ('rock', 200)
    PAPER_TARGET = ('paper', 200)
    SCISSORS_TARGET = ('scissors', 200)
    ROCK_TARGET_UPPERCASE = ('ROCK', 200)
    PAPER_TARGET_UPPERCASE = ('PAPER', 200)
    SCISSORS_TARGET_UPPERCASE = ('SCISSORS', 200)
    ROCK_TARGET_INCOMPLETE = ('roc', 500)
    ROCK_TARGET_SPACING = ('roc k', 500)
    ROCK_TARGET_NEWLINE = ('rock\n', 500)
    EMPTY_TARGET = ('', 500)
    WRONG_TARGET_1 = ('wrong_target_example_123', 500)
    WRONG_TARGET_2 = ('1234', 500)
    WRONG_TARGET_3 = ('\n', 500)

class ServerTestCase(unittest.TestCase):

    """
        Integration testing suite for the Flask server
        Calls the route functions
    """

    def test_api_v1_access_test(self):
        """
            Test the GET api/v1/ route of the REST server
        """
        output = api_v1_access_test()
        self.assertIs(output['status'], 200, 'AssertionIsError : api_v1_access_test status 500')
        assert output['route'] == 'api/v1/', 'AssertError : api_v1_access_test wrong route'
    
    def test_api_v1_generate(self):
        """
            Test the GET api/v1/generate?target=<string> route of the REST server
        """
        for target in Targets:
            # Put target enum item values in more readable vars
            input = target.value[0]
            expected_status = target.value[1]
            # Run api_v1_generate route
            output = api_v1_generate(target_arg=input)
            # Analyse output
            ### STATUS CHECK
            self.assertEqual(output['status'], expected_status, 'AssertionIsError : api_v1_generate status')
            ### ROUTE CHECK
            assert output['route'] == 'api/v1/generate', 'AssertError : api_v1_access_generate wrong route'
            ### IMAGE DATA CHECK
            try:
                im_b64 = output['data']['image_data']['value']
                im_bytes = base64.b64decode(im_b64) # decode b64 into binary format
                im_file = BytesIO(im_bytes) # convert image to file-like object
                try:
                    img = Image.open(im_file)
                    # TODO: Further checks here ?
                    pass
                except IOError:
                    self.fail('IOError : Image cannot be created from b64 encoded string')
            except KeyError:
                if expected_status == 500:
                    # When the server is expected to return status 500, 
                    # 'value' won't exist in the dict response
                    pass
                else:
                    self.fail('KeyError : Key not in output[\'data\'][\'image_data\'][\'value\']')
    
if __name__ == '__main__':
    unittest.main(verbosity=2)
