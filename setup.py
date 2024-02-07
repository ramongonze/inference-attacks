import setuptools

install_requires = [
    "numpy >= 1.21.5",
    "pyreadr",
    "pandas",
    "humanize",
    "matplotlib",
    "sas7bdat"
]

docs_require = [
    "sphinx >= 1.4",
    "sphinx-theme >= 1.0",
    "sphinx-rtd-theme >= 1.0"
]

setuptools.setup(
    name="infattacks",
    version="1.0",
    python_requires=">=3.8",
    description="Python library for Inference Attacks",
    author="Ramon Gonçalves Gonze",
    author_email="ramongonze@gmail.com",
    url="https://github.com/ramongonze/infattacks",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    extras_require={"docs": docs_require},
)
