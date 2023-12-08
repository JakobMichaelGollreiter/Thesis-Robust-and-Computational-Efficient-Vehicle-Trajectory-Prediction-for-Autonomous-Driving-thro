import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="commonroad-prediction",
    version="1.0.0",
    description="Additional prediction functions to CommonRoad",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['commonroad_prediction'],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

# EOF
