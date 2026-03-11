from setuptools import setup, find_packages

setup(
    name             = "ab_diagnostics",
    version          = "0.1.0",
    description      = "Lightweight A/B experiment diagnostics toolkit",
    packages         = find_packages(),
    python_requires  = ">=3.8",
    install_requires = [
        "numpy>=1.21",
        "pandas>=1.3",
        "scipy>=1.7",
    ],
)
