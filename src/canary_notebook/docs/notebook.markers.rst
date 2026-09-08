.. Copyright NTESS. See COPYRIGHT file for details.

   SPDX-License-Identifier: MIT

.. _notebook-markers:

Cell markers
============

Cell behavior is controlled by **comment markers** embedded in the cell source.
Markers use the syntax ``# [key: value]`` and must appear in comment lines.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Marker
     - Effect
   * - ``# [skip: true]``
     - Do not execute this cell at all.
   * - ``# [check_output: false]``
     - Execute the cell but do not compare its output against the stored reference.
   * - ``# [check_output: true]``
     - Force output comparison even when ``--notebook-dont-compare-outputs`` is set.
   * - ``# [allow_failure: true]``
     - An error raised by this cell does not fail the test; execution continues.
   * - ``# [raises: ExceptionType]``
     - This cell *must* raise the named exception.  Any other outcome fails the test.
   * - ``# [timeout: T]``
     - Per-cell execution timeout.  Accepts seconds (numeric) or duration strings
       such as ``5m`` or ``1h30m``.

Multiple markers can appear in consecutive comment lines at the top of a cell:

.. code-block:: python

   # [allow_failure: true]
   # [check_output: false]
   import something_optional

An unknown marker name produces a ``UserWarning``.


Examples
--------

Skip a cell entirely:

.. code-block:: python

   # [skip: true]
   expensive_setup_that_should_not_run_in_ci()

Mark non-deterministic output:

.. code-block:: python

   # [check_output: false]
   import datetime
   print(datetime.datetime.now())

Allow an optional dependency to be missing:

.. code-block:: python

   # [allow_failure: true]
   import optional_package

Assert that a cell raises a specific exception:

.. code-block:: python

   # [raises: ValueError]
   raise ValueError("expected error")

Set a per-cell timeout:

.. code-block:: python

   # [timeout: 5m]
   long_running_computation()
