import os
from setuptools import setup

setup(
    name="ReprDynamics",
    version="0.1",
    description="Visualization of representation dynamics",
    author="Noah",
    include_package_data=True,
    zip_safe=False,
    install_requires=['click==7.1.2',
        "click_logging",
        "matplotlib",
        'numpy',
        'opencv-python==4.5.2.52',
        'pandas==1.1.3',
        'pathlib==1.0.1',
        "Pillow",
        'protobuf==3.15.8',
        'scikit-image',
        'scikit-learn==0.24.2',
        "scipy",
        'tensorboard==2.3.0',
        'tensorboard-plugin-wit==1.7.0',
        'tensorboardX==2.1',
        'torch==1.8.1',
        "seaborn",
        "torchaudio",
        "torchvision",
        'tqdm==4.50.2',
        'recordclass',
        'flask'
      ],

    entry_points={
        "console_scripts": ["ReprDynamics = app:run"]
    },
)