import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="tencirchem-ng",
    version="2024.10",
    author="TenCirChem-NG Authors",
    author_email="liw31@g163.com",
    description="Efficient quantum computational chemistry based on TensorCircuit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tensorcircuit/TenCirChem-NG",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "tensorcircuit-ng[qiskit]",
        "pyscf",
        "openfermion",
        "pylatexenc",
        "noisyopt",
        "renormalizer",
    ],
    extras_require={
        "jax": ["jax", "jaxlib"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
