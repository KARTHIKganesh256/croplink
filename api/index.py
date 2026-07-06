import os
import sys

# Add root folder to sys.path so server can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app

# Expose app as the WSGI handler
app = app
