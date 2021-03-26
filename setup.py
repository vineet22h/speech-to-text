from setuptools import setup

setup(
    name='speech_to_text',
    version='1.0',
    author='Vineet Mukesh Haswani',
    author_email='vineethaswani2@gmail.com',
    install_requires=[
        "setuptools>=42",
        "wheel",
        "tensorflow==2.3.1",
        "pandas==1.0.1",
        "tqdm==4.42.1",
        "scikit-learn==0.23.2",
        "matplotlib==3.1.3",
        "seaborn==0.10.0",
        "numpy==1.16.5",
        "SoundFile",
    ],
)