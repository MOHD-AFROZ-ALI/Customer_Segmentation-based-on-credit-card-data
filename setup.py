from setuptools import setup, find_packages

setup(
    name="customer_segments_production",
    version="0.1.0",
    description="Customer Segmentation Project with clustering and Flask UI",
    author="MOHD AFROZ ALI",
    author_email="afrozali3001.aa@gmail.com",
    packages=find_packages(include=["custsegments", "custsegments.*"]),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "flask",
        "pyyaml",
        "dill",
        # Add other dependencies from your requirements.txt here
    ],
    python_requires='>=3.7',
)