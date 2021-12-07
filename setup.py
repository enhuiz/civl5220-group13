import subprocess
from pathlib import Path
from datetime import datetime
from setuptools import setup, find_packages


def shell(*args):
    out = subprocess.check_output(args)
    return out.decode("ascii").strip()


def write_version(version_core, pre_release=True):
    if pre_release:
        try:
            time = shell("git", "log", "-1", "--format=%cd", "--date=iso")
            time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S %z")
            time = time.strftime("%Y%m%d%H%M%S")
        except:
            time = "0" * 14
        version = f"{version_core}-dev{time}"
    else:
        version = version_core

    with open(Path("civl5220_group13", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))

    return version


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="civl5220_group13",
    python_requires=">=3.9.0",
    version=write_version("0.0.1", True),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["civl5220_group13"],
    install_requires=["argparse-node==0.0.2"],
    url="https://github.com/enhuiz/civl5220-group13",
    entry_points={
        "console_scripts": [
            "civl5220-group13=civl5220_group13:main",
        ],
    },
)
