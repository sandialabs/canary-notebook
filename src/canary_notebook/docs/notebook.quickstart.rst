.. Copyright NTESS. See COPYRIGHT file for details.

   SPDX-License-Identifier: MIT

.. _notebook-quickstart:

Quick start
===========

Once ``canary-notebook`` is installed, any ``.ipynb`` file on a ``canary run``
path is discovered and executed automatically.

.. code-block:: console

   # Run a single notebook
   canary run path/to/my_notebook.ipynb

   # Run all notebooks under a directory
   canary run path/to/notebooks/

   # Use the current Python environment's kernel
   canary run --notebook-current-env path/to/notebook.ipynb

   # Use a named kernel
   canary run --notebook-kernel-name python3 path/to/notebook.ipynb

   # Disable output comparison globally
   canary run --notebook-dont-compare-outputs path/to/notebook.ipynb

   # Set per-cell timeout (seconds)
   canary run --notebook-cell-timeout 120 path/to/notebook.ipynb


Execution model
---------------

* Each notebook produces **one canary test case**.
* Code cells are executed sequentially in one live kernel; state from earlier
  cells persists into later ones.
* Markdown and raw cells are skipped.
* ``.ipynb_checkpoints/`` directories are excluded from collection automatically.
* A test **fails** if any code cell raises an unhandled exception.  Execution
  continues past failing cells so all errors are visible in a single run.


CLI options
-----------

All options are in the ``canary notebook`` group and apply only to
``canary run``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Option
     - Description
   * - ``--notebook-config FILE``
     - YAML file with regex/replace pairs to sanitize cell outputs before comparison.
   * - ``--notebook-current-env``
     - Use the Python env that launched canary (ignores the kernel stored in the notebook).
   * - ``--notebook-kernel-name NAME``
     - Use a named Jupyter kernel.
   * - ``--notebook-cell-timeout T``
     - Maximum time in seconds for a single cell to execute.  Default: 2000.
   * - ``--notebook-kernel-startup-timeout T``
     - Maximum time in seconds to wait for the kernel to start.  Default: 60.
   * - ``--notebook-dont-compare-outputs``
     - Disable output comparison for all cells.

.. note::

   ``--notebook-current-env`` and ``--notebook-kernel-name`` are mutually
   exclusive.
