__author__ = 'jrl44'
__description__ = 'Listens for messages and sends an action in response'

import zmq
import time

import constants

SAFE_WORDS = ['stop', 'terminate', 'die', 'kill', 'pineapple', 'fluggaenkoecchicebolsen']

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect('tcp://localhost:%d' % constants.ZEROMQ_PORT)

print('Listening on port %d' % constants.ZEROMQ_PORT)

while True:
    message = socket.recv_json()
    if message in SAFE_WORDS:
        print('Received safe word')
        break
    else:
        print('Received message:\n\n%s\n' % message)
        print('Sleeping')
        time.sleep(0)
        learners = message['learners']
        print('Sending message back')
        socket.send_json(dict(selection=learners[0]))

print('Terminating')