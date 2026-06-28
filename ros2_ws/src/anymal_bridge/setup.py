from setuptools import find_packages, setup

setup(
    name="anymal_bridge",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    install_requires=["setuptools", "pyzmq"],
    entry_points={
        "console_scripts": [
            "bridge_node = anymal_bridge.bridge_node:main",
        ],
    },
)
