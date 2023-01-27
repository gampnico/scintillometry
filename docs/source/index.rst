.. Scintillometry Tools documentation master file, created by
   sphinx-quickstart on Fri Jan 27 12:10:29 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
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

Scintillometry Tools
====================

Analyse data & 2D flux footprints from Scintec's BLS scintillometers.

Running from Console
--------------------

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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro
   quickstart
   scintillometry
   scintillometry.wrangler
   scintillometry.backend
   scintillometry.metrics
   scintillometry.visuals

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
