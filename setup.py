from setuptools import setup, find_packages

with open('README.md') as f:
    README = f.read()

setup(
    name='smart-synapses',
    install_requires=[
        'numpy>=1.15<2.0',
        'scipy>=1.4<2.0',
    ],
    packages=find_packages(),
    version='0.0.1',
    author='Aleksei Sanin',
    author_email='vozhdb@gmail.com',
    description='',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://bitbucket.org/vogdb/synapse-plasticity-ann-snn/',
    classifiers=[]
)
