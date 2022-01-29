#!/usr/bin/env python3

import datetime

def response(status, route, data):
    """
        Function that returns a JSON-friendly Python dictionary 
        Used for the REST API responses
    """
    return {
        'status': status,
        'timestamp': datetime.datetime.now(), 
        'route': route,
        'data': data
    }