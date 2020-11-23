import setuptools

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setuptools.setup(
    name="SVARpy", # Replace with your own username
    version="0.1.0",
    author="Sascha Keweloh",
    author_email="sascha.keweloh@tu-dortmund.de",
    description="SVAR estimation",
    long_description=readme,
    url="https://github.com/Saschakew/SVARpy",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)