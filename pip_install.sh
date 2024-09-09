#!/bin/bash
set -e 

# Reminder for editable and compat modes. compat lets pylance find it if its not in site-packages
pip install -e . --config-settings editable_mode=compat