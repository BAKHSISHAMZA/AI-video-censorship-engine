from setuptools import setup, find_packages

setup(
    name="censorship_engine",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "ultralytics>=8.0.0",
    ],
    python_requires=">=3.8",
)