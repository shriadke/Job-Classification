import setuptools

with open("README.md", "r", encoding="utf-8") as rd:
    long_description = rd.read()


__version__ = "0.0.0"

REPO_NAME = "Job-Classification"
AUTHOR_USER_NAME = "shriadke"
SRC_REPO = "jobClassification"
AUTHOR_EMAIL = "shrinidhi.adke@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Job Classifications based on O*NETs",
    long_description=long_description,
    long_description_content ="text/markdown",
    url=f"https://www.github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug_tracker": f"https://www.github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)