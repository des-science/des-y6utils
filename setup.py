from setuptools import setup, find_packages

setup(
    name='des_y6utils',
    description="DES Y6 utilities",
    author="DES",
    packages=find_packages(),
    include_package_data=True,
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
)
