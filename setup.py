# The purpose of this setup.py is to make my project as a package and then we can deply it as a pipeline, so that we deply it on 
# server everybody can use it and install it.

from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT='-e .'

def get_requirements(filepath:str) -> List[str]:
    '''
    Docstring for get_requirements
    
    :param filepath: Description
    :type filepath: str
    :return: Description
    :rtype: List[str]
    this will get all the libraries to install in requirements.txt file
    '''
    requirements = []
    with open(filepath) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements
setup(
    name="mlproject",
    version="0.0.1",
    author="Kashif Karim",
    author_email="kashifkarimkhan88@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)