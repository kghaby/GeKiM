from setuptools import setup, find_packages

setup(
    name='GeKiM', 
    version='0.1.0', 
    author='Kyle Ghaby', 
    author_email='kyleghaby@gmail.com',  
    description='Generalized Kinetic Modeler: A Python package for modeling arbitrary kinetic schemes.',  
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',
    url='https://github.com/kghaby/GeKiM', 
    packages=find_packages(where="src"), 
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        'numpy', 
        'scipy', 
    ],
    classifiers=[
        'Development Status :: 3 - Alpha', 
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.8', 
)
