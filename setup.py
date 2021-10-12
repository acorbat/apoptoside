from setuptools import setup, find_packages

setup(
    name='apoptoside',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/acorbat/apoptoside/tree/master/',
    license='MIT',
    author='Agustin Corbat',
    author_email='acorbat@df.uba.ar',
    description='Package used for analysing homoFRET experiments with caspase '
                'biosensors.',
    install_requires=['matplotlib', 'numpy', 'pandas', 'pyDOE', 'scipy',
                      'seaborn',
                      'caspase_model @ git+https://github.com/acorbat/caspase_model.git']
)