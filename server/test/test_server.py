#!/usr/bin/env python3

import unittest
import signal
import os
import requests
from subprocess import Popen

class ServerTestCase(unittest.TestCase):

    """
        Integration testing suite for the Flask server
        Sends REST requests to the server running in localhost as a subprocess created in setUp()
    """

    def setUp(self):
        print("Creating server subprocess ...")
        self.server_subprocess_pid = -1
        self.server_subprocess_pid = (Popen(['python', '../app.py'])).pid
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
        # Target constants for requests
        ROCK_TARGET = 'rock'
        PAPER_TARGET = 'paper'
        SCISSORS_TARGET = 'scissors'
        EMPTY_TARGET = ''
        WRONG_TARGET_1 = 'wrong_target_example_123'
        WRONG_TARGET_2 = '1234'
        WRONG_TARGET_3 = '\n'
        # Send HTTP request to the ImageGen endpoint with a target parameter
        # response = requests.get("http://localhost:5050/api/v1/generate?target=")
        pass

    def tearDown(self):
        # Only kill subprocess if it is actually running
        if self.server_subprocess_pid != -1:
            print("Killing server subprocess ...")
            os.kill(self.server_subprocess_pid, signal.SIGKILL)
    
if __name__ == '__main__':
    unittest.main(verbosity=2)
