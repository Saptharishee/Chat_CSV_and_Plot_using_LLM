import re
import subprocess

# Path to your Streamlit app file
app_file_path = 'app.py'

def extract_libraries(file_path):
    libraries = set()
    with open(file_path, 'r') as file:
        content = file.read()
        # Regular expression to find import statements
        imports = re.findall(r'^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)', content, re.MULTILINE)
        for imp in imports:
            # Extract package names and add them to the set
            if '.' in imp:
                package = imp.split('.')[0]  # For 'package.subpackage'
            else:
                package = imp
            libraries.add(package)
    return libraries

def generate_requirements(libraries, output_file='requirements.txt'):
    with open(output_file, 'w') as file:
        for library in sorted(libraries):
            file.write(f'{library}\n')

def main():
    libraries = extract_libraries(app_file_path)
    generate_requirements(libraries)
    print(f'{app_file_path} processed. Requirements file generated.')

if __name__ == '__main__':
    main()
