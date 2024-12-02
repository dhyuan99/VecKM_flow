from setuptools import setup, find_packages

setup(
    name="VecKM_flow",
    version="0.1.0",
    author="Dehao Yuan",
    author_email="dhyuan@umd.edu",
    description="It is a point-based normal flow estimator from event camera inputs.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dhyuan99/VecKM_flow",  # Replace with your repo
    packages=find_packages(),  # Automatically finds sub-packages
    include_package_data=True,
    package_data={"VecKM_flow": ["models/*.pth"]},
    install_requires=[
        "torch>=1.13.0",
        "scipy==1.14.1",
        "scikit-learn==1.5.0",
        "tqdm==4.66.2",
        "matplotlib==3.8.3",
        "matplotlib-inline==0.1.7",
        "opencv-python==4.9.0.80"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)