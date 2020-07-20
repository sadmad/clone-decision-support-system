#!/usr/bin/python

import logging
import sys



logging.basicConfig(stream=sys.stderr)

if sys.version_info[0]<3:       # require python3
 raise Exception("Python3 required! Current (wrong) version: '%s'" % sys.version_info)
sys.path.insert(0, '/home/mdbadmin/decision-support-system/')
from run import app as application
application.secret_key = 'Add your secret key'