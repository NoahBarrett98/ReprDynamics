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
        'click==7.1.2',
        'numpy',
        'opencv-python==4.5.2.52',
        "Pillow",
        'scikit-image',
        'scikit-learn==0.24.2',
        "scipy",
        'torch==1.8.1',
        "torchvision",
        'flask'
      ],

    entry_points={
        "console_scripts": ["ReprDynamics = app:run"]
    },
)