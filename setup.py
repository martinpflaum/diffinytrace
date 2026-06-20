from setuptools import setup, find_packages
import os

here = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(here, 'requirements.txt')) as f:
    install_requires = f.read().splitlines()

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='diffinytrace',  # Package name
    version='2.4',  # Version number
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Martin Pflaum',
    author_email='contact@martinpflaum.com',
    classifiers=[],
    python_requires='==3.12',
)