import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autoder",
    version="0.0.2",
    author="Example Author",
    author_email="author@example.com",
    description="A small automatic differentiation package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AutoDiffAll/cs207_FinalProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
