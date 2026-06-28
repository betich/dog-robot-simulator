from setuptools import find_packages, setup

setup(
    name="rmf_adapter",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    install_requires=["setuptools"],
    entry_points={
        "console_scripts": [
            "anymal_fleet_adapter = rmf_adapter.anymal_fleet_adapter:main",
        ],
    },
)
