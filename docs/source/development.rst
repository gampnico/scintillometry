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

For Developers
==============

Install as Developer
--------------------

Contributions to the package's development must be compatible with Python 3.8+
and debian-based Linux distros. It is strongly recommended to use the provided
Makefile to duplicate workflows.

Install with development dependencies:

.. code-block:: bash

   git clone https://github.com/gampnico/scintillometry-tools.git
   make install-dev

If conda is unavailable:

.. code-block:: bash

   git clone https://github.com/gampnico/scintillometry-tools.git
   pip install -e .[dev]

With Python 3.9+, optionally install `Scalene`_ to profile your code.

.. _`Scalene`: https://github.com/plasma-umass/scalene

Contribution Guidelines
------------------------

Avoid excess conflicts by following these guidelines:

   - Write commit messages in the style of `conventional commits`_.
   - Write tests before committing features.
   - Push many small commits instead of a single large one.
   - Push new features to new branches. Never push to main.
   - Push documentation separately. Don't push built documentation.
   - Follow the `Google Style Guide`_.
   - Format all code with `black`_, line length 88.
   - Format *docstrings* with line length 72.
   - Format .rst files with 3-space indents, line length 80.
   - Break lines in Markdown only at the end of paragraphs.
   - Spaces, not tabs.

.. _`conventional commits`: https://www.conventionalcommits.org/en/v1.0.0/
.. _`Google Style Guide`: https://google.github.io/styleguide/pyguide.html
.. _`black`: https://black.readthedocs.io/en/stable/

Before Committing
-----------------

Please use the provided rc files for pylint and coverage. Ensure any changes are
covered by a relevant test. Always format and run tests with Python 3.8 before
committing:

.. code-block:: bash

   make tests

After adding changes with ``git add``, run:

.. code-block:: bash

   make commit

This formats code and runs tests before launching the commit dialogue. Write a
useful commit message:

.. code-block:: text

   feat(backend): compute bar with function foo

   Adds function foo that does bar.

   Begins:
           #ABC: select input for foo via cli
           #DEF: fix bug in foo

   Refs: #ABC, #DEF, #XYZ, ...

Finally push to the appropriate branch.
