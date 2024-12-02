from setuptools import setup, find_packages

setup(
    name="mist",
    version="0.1.1",
    author="Yuetian",
    author_email="yuetian.classic@outlook.com",
    description="A plug-and-play module for deploying MIST to LLM training",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Stry233/mist.llm",  # Repository URL
    packages=find_packages(),  # Automatically finds sub-packages
    install_requires=[
        "numpy>=1.19.2",
        "requests>=2.25.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
