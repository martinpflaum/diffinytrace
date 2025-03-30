from setuptools import setup, find_packages


with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='diffinytrace',  # Package name
    version='0.1',  # Version number
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=install_requires,
    author='Martin Pflaum',
    author_email='contact@martinpflaum.com',
    classifiers=[],
    python_requires='>=3.10',
)