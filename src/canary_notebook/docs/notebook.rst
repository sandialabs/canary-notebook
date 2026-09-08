.. Copyright NTESS. See COPYRIGHT file for details.

   SPDX-License-Identifier: MIT

.. _notebook-plugin:

Jupyter notebook testing (``canary-notebook``)
===============================================

.. note::

   ``canary-notebook`` is **not** a built-in canary extension.  It must be
   installed separately:

   .. code-block:: console

      pip install canary-notebook

   Source and issue tracker:
   `github.com/sandialabs/canary-notebook <https://github.com/sandialabs/canary-notebook>`_

The ``canary-notebook`` plugin enables `Jupyter notebooks`_ (``.ipynb`` files) to
be discovered and executed as first-class canary test cases.  Each notebook is
treated as a **single test case**.  All code cells are executed sequentially in
one live Jupyter kernel.  If any cell raises an unhandled exception the test
fails, but execution of subsequent cells continues so that all cell errors are
visible in one run.

.. _Jupyter notebooks: https://jupyter.org/

.. toctree::
   :maxdepth: 2

   notebook.install
   notebook.quickstart
   notebook.markers
   notebook.comparison
   notebook.config
