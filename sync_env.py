### This script exists to read the .env file and create a duplicate of the file at .env.sanitised
### without any sensitive values

### This script will be run as a pre-commit hook by Git

### This ensures the specific paths/secrets in .env are preserved, but the remote repo is kept
### up-to-date with a list of necessary global variables

import os

def sync_env_example():
    if not os.path.exists(".env"):
        return

    with open(".env", "r") as f_in, open(".env.sanitised", "w") as f_out:
        for line in f_in:
            # Preserve comments and empty lines
            if line.strip().startswith("#") or not line.strip():
                f_out.write(line)
            # Strip values from variables
            elif "=" in line:
                key = line.split("=", 1)[0]
                f_out.write(f'{key} = "insert_value_here"\n')

if __name__ == "__main__":
    sync_env_example()