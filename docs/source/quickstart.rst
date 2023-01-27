.. Copyright 2023 Nicolas Gampierakis.

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

Quick Start
===========

Installation
------------

Install with pip or conda:

``pip install -r requirements.txt .``

To build this documentation, ensure both ``sphinx`` and ``sphinx_rtd_theme`` are installed in your python environment. Navigate to docs, and run:

``make clean && make html``

To run tests or coverage, ensure ``pytest`` and ``pytest-dependency`` are installed in your Python environment, and run pytest from the package root.

Run from Terminal
-----------------

Usage: ``src/main.py [-h] [-i <input_data>] [-d] [...] [-v]``

Options and arguments (and corresponding environment variables):

Required arguments:
    ``-i, --input <path>: Path to raw BLS450 data.``

Optional switches:
    ``-h, --help: Show this help message and exit.``
    ``-z, --dry-run: Dry run of model.``
    ``-v, --verbose: Verbose mode.``

Optional arguments:
    ``-t, --timezone <str>: Convert to local timezone. Default "CET".``
    ``-c, --calibrate <float> <float>: Recalibrate path lengths.``

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
