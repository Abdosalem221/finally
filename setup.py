
from setuptools import setup, find_packages

setup(
    name="landmark",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'flask',
        'flask-sqlalchemy',
        'flask-migrate'
    ],
    entry_points={
        'flask.commands': [
            'run=app:create_app'
        ]
    }
)

setup(
    name="app",
    packages=find_packages(),
    version="0.1.0"
)
