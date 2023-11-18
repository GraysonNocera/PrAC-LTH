import subprocess

def main():

  with open("tests.txt", "r") as f:
    commands = f.readlines()

  for command in commands:  
    if not command: continue

    command = command.strip()
    if not command: continue

    print(f"Running command: {command}")
    retval = execute_command(command)
    if (retval.returncode != 0):
      print(f"Error running command {command}: {retval.stderr}")
    
def execute_command(command):
  args = command.split(' ')
  return subprocess.run(args, shell=True, capture_output=True)

main()