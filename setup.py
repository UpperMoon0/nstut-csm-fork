from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read the contents of README.md
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='nstut-csm-fork',
    version='0.1.2', 
    author='NsTut',
    author_email='nstut123@gmail.com',
    description='Fork of CSM for text-to-speech synthesis.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/UpperMoon0/nstut-csm-fork', 
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License', # Assuming MIT License based on typical open-source projects
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
    ],
    python_requires='>=3.10',
)
