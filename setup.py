import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='robust_cfe',
    version='0.0.1',
    author='Marco Virgolin',
    author_email='marco.virgolin@cwi.nl',
    url='https://github.com/marcovirgolin/robust_cfe',
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=['wheel'],
    install_requires=[
    	'numpy >= 1.16.1',
        'scikit-learn==1.0.1',
        'scipy',
        'joblib',
        'cma==3.1.0',
        'pandas',
	      'dice-ml',
	      'typing-extensions==4.2.0',
	      'fat-forensics',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
