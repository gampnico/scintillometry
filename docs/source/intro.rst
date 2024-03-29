.. Copyright 2023 Scintillometry Contributors.

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
   
Introduction
============

*Scintillometry* is configured for scintillometer experiments in Austria
using public or local data (ZAMG, InnFLUX), but is easily modified to work with
other data sources. Note that external data sources may have different licensing
constraints.

The package is currently in alpha and may change or break often. Support is only
available for Python 3.8+ on debian-based Linux distros.

This package started life as part of a field course several years ago. If you
spot any missing citations or licenses please contact me directly or open an
issue.

Features
--------

Scintillometry
**************

Data processing:
   - Parse scintillometry data from Scintec's BLS series of large aperture
     scintillometers (.mnd files).
   - Recalibrate data if the scintillometer was incorrectly set up (e.g. wrong
     dip switch settings).
   - Parse topographical data as path transect.
   - Parse meteorological data.
   - Parse innFLUX and HATPRO data.

Metrics:
   - Calculate effective path heights under various stability conditions.
   - Derive |CT2| values from |Cn2| if these were not collected.
   - Estimate the time where stability conditions change. 
   - Compute parameters such as Obukhov length, friction velocity, etc.
   - Compute kinematic and sensible SHF. Supports free convection and iteration
     with MOST: several sets of coefficients are available for MOST functions,
     based on previous studies.

Visualisation:
   - Produces time series of scintillometry and meteorological data.
   - Produces vertical profiles.
   - Produces plots for derived or iterated variables.
   - Produces comparisons between calculated parameters and external data
     sources.

Currently implemented MOST functions:
   - **an1988**: E.L. Andreas (1988) [#andreas1988]_, DOI:
     `10.1364/JOSAA.5.000481 <https://opg.optica.org/josaa/abstract.cfm?uri=josaa-5-4-481>`_
   - **li2012**: D. Li et al. (2012) [#li2012]_, DOI:
     `10.1007/s10546-011-9660-y <https://link.springer.com/article/10.1007/s10546-011-9660-y>`_
   - **wy1971**: Wyngaard et al. (1971) [#wyngaard1971]_, DOI:
     `10.1364/JOSA.61.001646 <https://opg.optica.org/josa/abstract.cfm?uri=josa-61-12-1646>`_
   - **wy1973**: Wyngaard et al. (1973) in Kooijmans and  Hartogensis (2016)
     [#kooijmans2016]_, DOI:
     `10.1007/s10546-016-0152-y <https://link.springer.com/article/10.1007/s10546-016-0152-y>`_
   - **ma2014**: Maronga et al. (2014) [#maronga2014]_, DOI:
     `10.1007/s10546-014-9955-x <https://link.springer.com/article/10.1007/s10546-014-9955-x>`_
   - **br2014**: Braam et al. (2014) [#braam2014]_, DOI:
     `10.1007/s10546-014-9938-y <https://link.springer.com/article/10.1007/s10546-014-9938-y>`_

Footprint Climatology (Roadmap)
*******************************

These features are under development.

Metrics:
   - Process 2D flux footprints generated by Natascha Kljun's online model,
     available here_.
   - Adjust topography and stitch footprints together.

.. _here: http://footprint.kljun.net/

Visualisation:
   - Produce regression plots between calculated parameters and external data
     sources.
   - Overlay stitched footprints onto map.

Example Workflow (Terminal)
---------------------------

This package supports SRun and Austrian-sourced data (ZAMG, InnFLUX)
out-of-the-box. If your scintillometry readings were taken in Austria, use
`DGM 5m data`_ to generate topographical data for the scintillometer's path
coordinates. Then generate the path transects necessary for calibrating the
scintillometer.

.. _`DGM 5m data`: https://www.data.gv.at/katalog/dataset/digitales-gelandemodell-des-landes-salzburg-5m

**Scintillometer path coordinates must be accurate. Incorrectly generated
topographical data leads to poor calibration and nonsense results!**

To install, follow the instructions given :doc:`here </quickstart>`.

Once installed, list all available arguments with:

.. code-block:: bash
    
   python3 ./src/scintillometry/main.py -h
   make commands  # if you prefer less typing

Navigate to the package root in the terminal. Calculate and plot surface
sensible heat fluxes using MOST in CET, with coefficients from Andreas (1988)
[#andreas1988]_:

.. code-block:: bash
    
   python3 ./src/scintillometry/main.py -i "./<path_to_input>/<bls_data>.mnd" \
   -p "./<path_to_transect>/<transect_data>.csv" -t "CET"

If you are not using the scintillometer in Austria, you will need to find and
parse topographical data yourself. Add or modify functions in
``wrangler/data_parsing.py`` to parse data from other scintillometers,
organisations, or countries.

Acknowledgements
----------------

This project would not be possible without invaluable contributions from
Dr. Manuela Lehner, Dr. Helen Ward, and Josef Zink.

References
----------

.. rubric:: References

.. [#andreas1988] |andreas1988|
.. [#braam2014] |braam2014|
.. [#kooijmans2016] |kooijmans2016|
.. [#li2012] |li2012|
.. [#maronga2014] |maronga2014|
.. [#wyngaard1971] |wyngaard1971|