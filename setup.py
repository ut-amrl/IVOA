from setuptools import setup, find_packages

setup(
    name='ivoa',
    version='1.0.0',
    description='IVOA.',
    packages=find_packages(include=['failure_detection', 'failure_detection.*']),
    install_requires=[
        'numpy',
        'scikit-image',
        'matplotlib'
    ]
)
