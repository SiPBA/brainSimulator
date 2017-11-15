from setuptools import setup

setup(
    name="brainSimulator",
    version="0.5.1",
    description="Nuclear brain imaging synthesis with python",
    author="SIPBA@UGR",
    author_email="sipba@ugr.es",
    url='https://github.com/SiPBA/brainSimulator',
    license="GPL-3.0+",
    py_modules=["brainSimulator"],
    install_requires=[
        "numpy",
		  "scikit-learn",
		  "scipy",
    ],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GPL-3.0+',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    # ...
)
