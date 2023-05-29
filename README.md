<!-- Copyright 2023 Scintillometry Contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# Scintillometry

[![Pytest and Flake8](https://github.com/gampnico/scintillometry/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/gampnico/scintillometry/actions/workflows/python-app.yml)

Development branch: [![Pytest and Linting](https://github.com/gampnico/scintillometry/actions/workflows/python-app.yml/badge.svg?branch=develop)](https://github.com/gampnico/scintillometry/actions/workflows/python-app.yml)

Analyse data & 2D flux footprints from Scintec's BLS scintillometers.

This project started life as part of a field course. If you spot any missing citations or licences please [open an issue](https://github.com/gampnico/scintillometry/issues).

Comprehensive documentation is available [via ReadTheDocs](https://scintillometry.readthedocs.io/en/latest/).

# Processing Scintillometry Data in Complex Terrain

*Scintillometry* is configured for scintillometer experiments in Austria using public or local data (ZAMG, InnFLUX), but is easily modified to work with other data sources. Note that external data sources may have different licensing constraints.

The package is currently in alpha and may change or break often. Support is only available for Python 3.8+ on debian-based Linux distros.

## Installation

*Scintillometry* supports package installation with pip, or from source as an editable with pip or conda. **Installing as an editable is recommended**, as it allows you to call command line arguments directly on the package instead of writing a frontend.

### Install from Source with Conda/Mamba

Create or activate your preferred conda environment and run:

```bash
git clone https://github.com/gampnico/scintillometry.git
make install
```

That's it!

Install the package with optional dependencies:

```bash
make install-tests  # install with dependencies for tests
make install-docs   # install and build local documentation
make install-all    # install with tests and build local documentation
make install-dev    # install with dependencies for development
```

Installation uses conda if mamba is unavailable. Micromamba may also work, but is not currently supported.

### Install with Pip

If conda/mamba are not your package managers, then run:

```bash
python3 -m pip install --upgrade scintillometry
```

Note that this installation method does not provide documentation or a Makefile, and you cannot easily use the package from the command line.

To install from source (recommended):

```bash
git clone https://github.com/gampnico/scintillometry.git
pip install -e .
```

Install the package with optional dependencies:

```bash
pip install -e .[tests]
pip install -e .[docs]
pip install -e .[tests,docs]    # no whitespace after comma
pip install -e .[dev]
```

# Features

## Scintillometry

Data processing:
- Parse scintillometry data from Scintec's BLS series of large aperture scintillometers (.mnd files).
- Recalibrate data if the scintillometer was incorrectly set up (e.g. wrong dip switch settings).
- Parse topographical data as path transect.
- Parse meteorological data.
- Parse innFLUX and HATPRO data.

Metrics:
- Calculate effective path heights under various stability conditions.
- Derive C<sub>T</sub><sup>2</sup> values from C<sub>n</sub><sup>2</sup> if these were not collected.
- Estimate the time when stability conditions change.
- Estimate the boundary layer height.
- Compute parameters such as Obukhov length, friction velocity, etc.
- Compute kinematic and sensible heat fluxes. Supports free convection and iteration with MOST: several sets of coefficients are available for MOST functions, based on previous studies.

Visualisation:
- Produces time series of scintillometry and meteorological data.
- Produces vertical profiles.
- Produces plots for derived or iterated variables.
- Produces comparisons between calculated parameters and external data sources.

Currently implemented MOST functions:
- **an1988**: E.L. Andreas (1988), [DOI: 10.1364/JOSAA.5.000481](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-5-4-481)
- **li2012**: D. Li et al. (2012), [DOI:10.1007/s10546-011-9660-y](https://link.springer.com/article/10.1007/s10546-011-9660-y)
- **wy1971**: Wyngaard et al. (1971), [DOI: 10.1364/JOSA.61.001646](https://opg.optica.org/josa/abstract.cfm?uri=josa-61-12-1646)
- **wy1973**: Wyngaard et al. (1973) in Kooijmans and  Hartogensis (2016), [DOI: 10.1007/s10546-016-0152-y](https://link.springer.com/article/10.1007/s10546-016-0152-y)
- **ma2014**: Maronga et al. (2014), [DOI: 10.1007/s10546-014-9955-x](https://link.springer.com/article/10.1007/s10546-014-9955-x)
- **br2014**: Braam et al. (2014), [DOI: 10.1007/s10546-014-9938-y](https://link.springer.com/article/10.1007/s10546-014-9938-y)

## Footprint Climatology (Roadmap)

These features are under development.

Metrics:
- Process 2D flux footprints generated by Natascha Kljun's online model, available [here](http://footprint.kljun.net/).
- Adjust topography and stitch footprints together.

Visualisation:
- Produce regression plots between calculated parameters and external data sources.
- Overlay stitched footprints onto map.

## Workflow

### Example Workflow

This package supports SRun and Austrian-sourced data (ZAMG, InnFLUX) out-of-the-box. If your scintillometry readings were taken in Austria, use [DGM 5m data](https://www.data.gv.at/katalog/dataset/digitales-gelandemodell-des-landes-salzburg-5m) to generate topographical data for the scintillometer's path coordinates. Then generate the path transects necessary for calibrating the scintillometer.

**Scintillometer path coordinates must be accurate. Incorrectly generated topographical data leads to poor calibration and nonsense results!**

List all available arguments with:

```bash
python3 ./src/scintillometry/main.py -h
make commands   # if you installed from source
```

Navigate to the package root in the terminal. Calculate and plot surface
sensible heat fluxes using MOST in CET, with coefficients from Andreas (1988):

```bash
python3 ./src/scintillometry/main.py -i "./<path_to_input>/<bls_data>.mnd" \
-p "./<path_to_transect>/<transect_data>.csv" -t "CET"
```

If you are not using the scintillometer in Austria, you will need to find and parse topographical and meteorological data yourself. Add parsing functions to classes in ``wrangler/data_parsing.py`` to parse data from other scintillometers, organisations, or countries. A step-by-step guide on how to do this is given in the module's docstring.

### Make Things Simple

If you installed from source, the provided Makefile has many uses. View all the available commands:

```bash
make help       # display help for Makefile targets
make commands   # display help for scintillometry
```

### Run from Terminal

From the package root, pass arguments like so:

```bash
src/scintillometry/main.py [-h] [-i <input_data>] [-p <path_data>] [-d] [...] [-v]
```

### Import as Package

*Scintillometry* and its submodules can be imported as Python modules:

```python
from scintillometry.wrangler.data_parser import WranglerParsing

parser = WranglerParsing()
dataframe = parser.scintillometer.parse_scintillometer(file_path="./data.mnd")
weather = parser.weather.parse_weather(file_path="./weather.csv")
weather = parser.weather.transform.change_index_frequency(weather, "60S")
```

MOST functions are stored in their respective class:

```python
from scintillometry.backend.iterations import IterationMost

iteration = IterationMost()
iteration.most_method(dataframe, eff_h, stability, coeff_id="an1988")
```

Many classes initialise atmospheric constants using the AtmosConstants class:

```python
from scintillometry.metrics.calculations import MetricsWorkflow
from scintillometry.backend.constants import AtmosConstants

workflow = MetricsWorkflow()
kelvin = workflow.constants.kelvin  # 273.15
assert isinstance(workflow.constants, AtmosConstants)  # True
```

For more information see the API section of the documentation.

# Acknowledgements

This project would not be possible without the invaluable contributions from Josef Zink, Dr. Manuela Lehner, and Dr. Helen Ward.
