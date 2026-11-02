"""
Local installation (alternative to the Dev Container).

From the repo root, inside a Python 3.9-3.11 virtual environment (3.11 recommended):

    pip install -e .

This installs the dependencies below and makes everything under `src/` importable
from any notebook, e.g. `from src.agents.dqn import DDQNAgent`.
"""
from setuptools import find_packages, setup

setup(
    name="best-execution",
    version="1.0.0",
    description="Execution algorithms with reinforcement learning - BME Institute (MIAX)",
    author="Jesús Sanz del Real",
    # TensorFlow 2.15 only ships wheels for Python 3.9-3.11
    # (3.9.0 and 3.9.1 are excluded by cryptography, see below)
    python_requires=">=3.9.2,<3.12",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        # Last TF with Keras 2, the API src/agents/dqn.py is written for (TF >= 2.16 uses Keras 3)
        "tensorflow==2.15.1",
        "numpy==1.26.4",  # TF 2.15 requires numpy < 2
        # data/*.pkl were pickled with pandas 1.x and pickle.load fails on them from pandas 2.1 on
        "pandas==2.0.3",
        "matplotlib==3.9.4",  # last series supporting Python 3.9
        "jupyterlab==4.5.11",  # last 4.x supporting Python 3.9
        # Transitive deps whose newer releases dropped macOS Intel wheels (would compile from source)
        "argon2-cffi-bindings==25.1.0",  # via jupyterlab -> jupyter-server -> argon2-cffi
        "cryptography==48.0.1",  # via tensorflow -> tensorboard -> google-auth
    ],
)
