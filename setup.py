import os
from setuptools import setup

setup(
    name="ReprDynamics",
    version="0.1",
    description="Visualization of representation dynamics",
    author="Noah",
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        
    ],
    entry_points={
        "console_scripts": ["ReprDynamics = app:run"]
    },
)