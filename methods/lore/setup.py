import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='lore',
    version='0.0.1',
    author='Riccardo Guidotti',
    author_email='',
    url='https://github.com/ricotti/lore',
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=['wheel'],
    install_requires=['numpy', 'scikit-learn', 'deap', 'pydot', 'pydotplus', 'imblearn', 'networkx'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
