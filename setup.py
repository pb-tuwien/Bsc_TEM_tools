from setuptools import setup, find_packages

# function to read the requirements from a file
def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

setup(
    name='Geophysics',
    version='0.1',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),  # Dependencies from requirements.txt
)
