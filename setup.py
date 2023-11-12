from setuptools import setup, find_packages

setup(
    name="language_agents_effectiveness",
    version="1.0.0",
    description="A description of your project",
    author="Andres Torres",
    author_email="andres@cloudadventures.net",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'setuptools',
    ],
    python_requires='>=3.10',
)