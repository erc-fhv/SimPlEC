from setuptools import setup, find_packages

# Read the contents of your README file
# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setup(
    name="SimPlE",  # Replace with your own package name
    version="0.0.1",  # Initial release version
    author="Valentin Seiler",
    author_email="valentin.seiler@fhv.at",
    description="Simulation Playground for Energy (time discrete and evend discrete cosimulaton)",
    long_description='',  # long_description,
    # long_description_content_type="text/markdown",  # Use README.md as long_description
    # url="https://github.com/your_username/your_repo_name",  # Project URL
    packages=find_packages(),  # Automatically find packages in the directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',  # Specify Python version requirements
    install_requires=[
        "numpy",  # List your dependencies here
        "networkx",
        "matplotlib",
        "pandas",
        "tqdm",
    ],
)