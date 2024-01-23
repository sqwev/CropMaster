import os
import datetime as dt
import numpy as np
import fiona
import json
import subprocess
import shutil

def execute_system_command(command: str, max_tries: int):
    if max_tries < 1:
        print("Wrong max_tries!")
        return False

    for i in range(max_tries):
        result = subprocess.run(command, shell=True, capture_output=True)

        if result.returncode == 0:
            print(f"Command succeeded on try {i + 1}")
            return result.stdout.decode('utf-8')
        else:
            print(f"Command failed on try {i + 1}:")
            print(result.stderr.decode('utf-8'))

    print(f"Command failed after {max_tries} tries!")
    return False

