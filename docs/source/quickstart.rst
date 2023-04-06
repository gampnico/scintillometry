.. Copyright 2023 Scintillometry-Tools Contributors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   =====

Using the Tools
===============

Installation
------------

Scintillometry-Tools supports installation with conda, mamba, and pip.

Install with Conda/Mamba
************************

Create or activate your preferred conda environment then run:

.. code-block:: bash

   git clone https://github.com/gampnico/scintillometry-tools.git
   make install

Install the package with optional dependencies:

.. code-block:: bash

   make install-tests   # install with dependencies for tests
   make install-docs    # install and build local documentation
   make install-all     # install with tests and build local documentation
   make install-dev     # install with dependencies for development

Installation uses conda if mamba is unavailable. Micromamba may also work, but is not currently supported. Whilst dependencies are installed with conda/mamba, the scintillometry-tools package is installed as an editable with pip.

Install with Pip
****************

If conda/mamba are not your package managers, then run:

.. code-block:: bash

   git clone https://github.com/gampnico/scintillometry-tools.git
   pip install -e .

Install the package with optional dependencies:

.. code-block:: bash

   pip install -e .[tests]       # install with dependencies for tests
   pip install -e .[docs]        # install with dependencies for documentation
   pip install -e .[tests,docs]  # no whitespace after comma
   pip install -e .[dev]         # install with dependencies for development

Run from Terminal
-----------------

View all command-line arguments:

.. code-block:: bash

   python3 ./src/scintillometry/main.py -h
   make commands  # if you prefer less typing

Usage:

.. code-block:: bash
   
   src/scintillometry/main.py [-h] [-i <input_data>] [-p <path_data>] [-d] [...] [-v]

Required arguments:
   -i, --input <path>      Path to raw BLS data.
   -t, --transect <path>       Path to topographical path transect.

Optional switches:
   -h, --help                 Show this help message and exit.
   -q, --specific-humidity    Derive fluxes from specific humidity.
   -z, --dry-run              Dry run of model.
   -v, --verbose              Verbose mode.

Optional arguments:
   -e, --eddy <str>                 Path to eddy covariance data (InnFLUX).
   -p, --profile <str>              Path to temperature and humidity profiles
                                       (HATPRO).
   -t, --timezone <str>             Convert to local timezone. Default "CET".
   -c, --calibrate <float float>    Recalibrate path lengths.
   -s, --stability <str>            Set default stability condition.
   -s, --switch-time <str>          Override local time of switch between
                                       stability regimes.
   -k, --station-id <str>           ZAMG station ID (Klima-ID). Default 11803.
   --location <str>                 Location of experiment. Overrides any other
                                       location metadata.
   --beam-wavelength <int>          Transmitter beam wavelength, nm.
                                       Default 850 nm.
   --beam-error <int>               Transmitter beam wavelength error, nm.
                                       Default 20 nm.

Import as Package
-----------------

Scintillometry-Tools and its submodules can be imported as a Python module:

.. code-block:: python

   import scintillometry
   from scintillometry.wrangler.data_parser import parse_scintillometer

MOST functions are stored in their respective class:

.. code-block:: python

   from scintillometry.backend.iterations import IterationMost

   workflow = IterationMost()
   workflow.most_method(dataframe, eff_h, stability, coeff_id="an1988")

These classes inherit from the AtmosConstants class:

.. code-block:: python
   
   from scintillometry.backend.constants import AtmosConstants

   constants = AtmosConstants()
   kelvin = constants.kelvin  # 273.15

Make Things Simple
------------------

The provided Makefile adds shortcuts for more complex commands. View all the available shortcuts:

.. code-block:: bash

   make help

Available shortcuts:
   :help:            Display this help screen.
   :install:         Install with conda.
   :install-tests:   Install with dependencies for tests.
   :install-docs:    Install with local documentation.
   :install-all:     Install package with tests & documentation.
   :install-dev:     Install in development mode.
   :commands:        Display help for scintillometry package.
   :test:            Format code and run tests.
   :doc:             Build documentation.
   :format:          Format all python files.
   :coverage:        Run pytest with coverage.
   :flake8:          Lint with flake8.
   :pylint:          Lint with Pylint.
   :scalene:         Profile with scalene (Python 3.9+).
   :black:           Format all python files with black.
   :isort:           Optimise python imports.
   :run:             Alias for `make commands`.
   :pkg:             Run test, build documentation, build package.
   :commit:          Format, test, then commit.
   
Some of these shortcuts will only work if the optional dependencies were installed.

Run Tests
---------

Install dependencies for tests:

.. code-block:: bash

   make install-tests

Run tests with coverage from the package root:

.. code-block:: bash

   make tests

Logs are placed in the ``./logs/`` folder.

Build Local Documentation
-------------------------

Install dependencies for documentation:

.. code-block:: bash

   make install-docs

Build the documentation:

.. code-block:: bash

   make docs

Formatting breaks if ``sphinx_rtd_theme`` version is less than 1.1.
