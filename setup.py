from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='GeKiM', 
    version='0.5.4', 
    author='Kyle Ghaby', 
    author_email='kyleghaby@gmail.com',  
    description='Generalized Kinetic Modeler: A Python package for modeling arbitrary kinetic schemes.',  
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',
    url='https://github.com/kghaby/GeKiM', 
    packages=find_packages(), 
    include_package_data=True,
    install_requires=required,
    classifiers=[
        'Development Status :: 4 - Beta', 
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.9', 
)
