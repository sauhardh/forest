from setuptools import setup, find_packages

setup(
    name="forest",
    version="0.1.0",
    description="Acoustic monitoring and bird species classification in Nepal",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    python_requires=">=3.10",
)
