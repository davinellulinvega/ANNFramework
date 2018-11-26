#!/usr/bin/python3
import os

# Applied twice dirname gives the path to the parent folder.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Declare the networks architectures.
NB_ACTIONS = 2

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': "%(asctime)s %(filename)s - %(levelname)s in %(module)s.%(funcName)s at line %(lineno)d:"
                      " %(message)s"
        }
    },
    'handlers': {
        'brain': {
            'level': 'WARNING',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, "logs", "brain.log"),
            'formatter': 'simple'
        },
        'brain_les': {
            'level': 'WARNING',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, "logs", "brain_les.log"),
            'formatter': 'simple'
        }
    },
    'loggers': {
        'Brain': {
            'handlers': ['brain'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'BrainLes': {
            'handlers': ['brain_les'],
            'level': 'DEBUG',
            'propagate': True,
        }
    },
}
