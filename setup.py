import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'networkx',
    'numpy',
    'scikit-learn',
    'matplotlib',
    'tensorflow>=1.12.0'
]

setuptools.setup(

    name="gnn",

    version="0.0.0",

    author="Weichen Shen",

    author_email="weichenswc@163.com",

    url="https://github.com/shenweichen/GraphNeuralNetwork",

    packages=setuptools.find_packages(exclude=[]),

    python_requires='>=3.5',  # 3.4.6

    install_requires=REQUIRED_PACKAGES,

    extras_require={

        "cpu": ['tensorflow>=1.12.0'],

        "gpu": ['tensorflow-gpu>=1.12.0'],

    },

    entry_points={

    },
    license="MIT license",

)
