import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='growingspheres',
    version='0.0.1',
    author='Thibault Laugel',
    author_email='',
    url='https://github.com/thibaultlaugel/growingspheres',
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=['wheel'],
    install_requires=[
    	'numpy >= 1.16.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
