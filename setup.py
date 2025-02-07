from setuptools import find_packages, setup
from typing import List


SETUP_STRING = '-e .'  # autometically enable or trigger the setup .py

def get_requirements(filename: str) -> List[str]:
    ''' 
        -> for the return type
        this function will return the list of all the dependencies that are needed for this project installation...
    '''
    requirements = []
    with open(filename, 'r') as file:
        dependencies = file.readlines()
        # print(dependencies)
        requirements = [dependency.strip() for dependency in dependencies]
        
        if SETUP_STRING in requirements:
            requirements.remove(SETUP_STRING)
        
        print(requirements)
        return requirements
        
        
        
setup(
    name='ML Project',
    version='0.0.1',
    author='Shimul Zahan',
    author_email='shimulzahan636@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)

''' 
    (function) def setup(
    *,
    name: str = ...,
    version: str = ...,
    description: str = ...,
    long_description: str = ...,
    long_description_content_type: str = ...,
    author: str = ...,
    author_email: str = ...,
    maintainer: str = ...,
    maintainer_email: str = ...,
    url: str = ...,
    download_url: str = ...,
    packages: list[str] = ...,
    py_modules: list[str] = ...,
     scripts: list[str] = ...,
    ext_modules: Sequence[Extension] = ...,
    classifiers: list[str] = ...,
    distclass: type[Distribution] = ...,
    script_name: str = ...,
    script_args: list[str] = ...,
    options: Mapping[str, Incomplete] = ...,
    license: str = ...,
    keywords: list[str] | str = ...,
    platforms: list[str] | str = ...,
    cmdclass: Mapping[str, type[Command]] = ...,
    data_files: list[tuple[str, list[str]]] =
    options: Mapping[str, Incomplete] = ...,
    license: str = ...,
    keywords: list[str] | str = ...,
    platforms: list[str] | str = ...,
    cmdclass: Mapping[str, type[Command]] = ...,
    data_files: list[tuple[str, list[str]]] =
     package_dir: Mapping[str, str] = ...,
    obsoletes: list[str] = ...,
    provides: list[str] = ...,
    requires: list[str] = ...,
    command_packages: list[str] = ...,
    command_options: Mapping[str, Mapping[str, tuple[Incomplete, Incomplete]]] = ...,
     package_data: Mapping[str, list[str]] = ...,
    include_package_data: bool = ...,
    libraries: list[tuple[str, _BuildInfo]] = ...,
    headers: list[str] = ...,
    ext_package: str = ...,
    include_dirs: list[str] = ...,
    password: str = ...,
    fullname: str = ...,
    **attrs: Any
) -> Distribution
'''