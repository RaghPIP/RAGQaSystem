from setuptools import setup, find_packages

setup(
    name="QAWithPDF",
    version="0.1.0",
    author="Ragotma Ragavendar",
    author_email="your_email@example.com",
    packages=find_packages(),
    install_requires=[
        "llama-index",
        "openai",
        "python-dotenv"
    ],
)
