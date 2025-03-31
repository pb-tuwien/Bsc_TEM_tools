from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="TEM_tools",
    version="0.1",
    packages=find_packages(include=["TEM_tools", "TEM_tools.*"]), # 
    install_requires=requirements,  # Use the requirements from requirements.txt
    description="A package for the processing of TEM data",
    long_description=open("README.md").read(),
    author="Peter Balogh, Jakob Welkens",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11,<3.12",
)