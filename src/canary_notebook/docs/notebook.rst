.. Copyright NTESS. See COPYRIGHT file for details.

   SPDX-License-Identifier: MIT

.. _notebook-plugin:

Jupyter notebook testing (``canary-notebook``)
===============================================

The ``canary-notebook`` plugin enables `Jupyter notebooks`_ (``.ipynb`` files) to
be discovered and executed as first-class canary test cases.  Each notebook is
treated as a **single test case**.  All code cells are executed sequentially in
one live Jupyter kernel.  If any cell raises an unhandled exception the test
fails, but execution of subsequent cells continues so that all cell errors are
visible in one run.

The plugin is independently distributed as the ``canary-notebook`` package:

.. code-block:: console

   pip install canary-notebook

.. _Jupyter notebooks: https://jupyter.org/

.. toctree::
   :maxdepth: 2

   notebook.install
   notebook.quickstart
   notebook.markers
   notebook.comparison
   notebook.config
