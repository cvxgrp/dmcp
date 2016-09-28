from setuptools import setup

setup(
    name='dmcp',
    version='0.1.3',
    author='Xinyue Shen, Steven Diamond, Stephen Boyd',
    author_email='xinyues@stanford.edu, diamond@cs.stanford.edu, boyd@stanford.edu',
    packages=['dmcp'],
    license='GPLv3',
    zip_safe=False,
    install_requires=["cvxpy >= 0.3.5"],
    use_2to3=True,
    url='http://github.com/cvxgrp/dmcp/',
    description='A CVXPY extension for multi-convex programs.',
)
