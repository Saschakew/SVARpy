import setuptools
from pathlib import Path

requirements_file = Path(__file__).parent / 'requirements.txt'
with requirements_file.open() as f:
    install_requires = f.read().splitlines()

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setuptools.setup(
    name="SVARpy",
    version="0.1.7",
    author="Sascha Keweloh",
    author_email="sascha.keweloh@tu-dortmund.de",
    description="SVAR estimation",
    long_description=readme,
    url="https://github.com/Saschakew/SVARpy",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires='>=3.7',
)
