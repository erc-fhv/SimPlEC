from setuptools import setup, find_packages

# Read the contents of your README file
# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setup(
    name="simplec",
    version="0.0.1",
    author="Valentin Seiler",
    author_email="valentin.seiler@fhv.at",
    description="Simulation Playground for Energy Communities (time discrete and evend discrete cosimulaton)",
    long_description='',
    # long_description_content_type="text/markdown",  # Use README.md as long_description
    # url="https://github.com/your_username/your_repo_name",  # Project URL
    packages=['simplec_examples'],  # find_packages(),  # Automatically find packages in the directory
    py_modules=['simplec'],
    python_requires='>=3.10',  # Specify Python version requirements
    install_requires=[
        "numpy",
        "networkx",
        "matplotlib",
        "pandas",
        "tqdm",
        "scipy",
        "pvlib"
    ],
)