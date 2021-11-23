from setuptools import setup, find_packages


def find_package_data():
    package_data = {
        "randomcarbon.data": [
            "templates/*"
        ]
    }
    return package_data


setup(
    name='randomcarbon',
    version='0.1',
    packages=find_packages(),
    package_data=find_package_data(),
    url='',
    license='',
    author='',
    author_email='',
    description='',
    setup_requires=['numpy', 'pymatgen', 'ase', 'pymongo']
)
