#!/usr/bin/env python3

from enum import Enum
import unittest
import signal
import os
import requests
from subprocess import Popen

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
        Sends REST requests to the server running in localhost as a subprocess created in setUp()
    """

    def setUp(self):
        print("Creating server subprocess ...")
        # Running server from root
        self.server_subprocess_pid = -1
        self.server_subprocess_pid = (Popen(['python', 'server/app.py'])).pid
        # Check whether subprocess failed, terminate test run if so
        if self.server_subprocess_pid == -1:
            self.fail('Server subprocess failed to create.')

    def test_api_v1_access_test(self):
        """
            Test the GET api/v1/ route of the REST server
        """
        # Send HTTP request
        response = requests.get("http://localhost:5050/api/v1/")
        # Parse HTTP response to JSON
        data = response.json()
        # Check status & data in response
        try:
            self.assertIs(data['status'], 200) # REST status
            assert(data['route'], 'api/v1/') 
        except AttributeError:
            self.fail('AttributeError : Returned data has missing response fields. Might mean the server is iresponsive.')
    
    def test_api_v1_generate(self):
        """
            Test the GET api/v1/generate?target=<string> route of the REST server
        """
        for target in Targets:
            # Send HTTP requests to the ImageGen endpoint with the target parameters, 
            # sequentially iterating through the Targets enum
            url_string = f'http://localhost:5050/api/v1/generate?target={target.value[0]}'
            response = requests.get(url_string)
            response_json = response.json()
            print(response_json)
            # Analyse response from request
            self.assertIs(response_json['status'], target.value[1])

    def tearDown(self):
        # Only kill subprocess if it is actually running
        if self.server_subprocess_pid != -1:
            print("Killing server subprocess ...")
            os.kill(self.server_subprocess_pid, signal.SIGKILL)
    
if __name__ == '__main__':
    unittest.main(verbosity=2)
