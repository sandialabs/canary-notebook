.. Copyright NTESS. See COPYRIGHT file for details.

   SPDX-License-Identifier: MIT

.. _notebook-config:

Configuration
=============

Sanitize config file
--------------------

Pass a YAML config file to ``--notebook-config`` to normalize non-deterministic
output before comparison.  The file must contain a ``notebook.sanitize`` list of
``regex``/``replace`` pairs:

.. code-block:: yaml

   # sanitize.yaml
   notebook:
     sanitize:
       - regex: '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
         replace: 'TIMESTAMP'
       - regex: '0x[0-9a-fA-F]+'
         replace: '0xADDR'
       - regex: '\d+\.\d+s'
         replace: 'X.XXs'

Then pass it to canary:

.. code-block:: console

   canary run --notebook-config sanitize.yaml path/to/notebook.ipynb

The patterns are applied to every string value in cell outputs before
comparison.  A sample config normalizing common non-deterministic values is
provided at ``sample_notebooks/sanitize_defaults.yaml`` in the
``canary-notebook`` source tree.


Kernel selection
----------------

The Jupyter kernel is selected in the following priority order:

1. ``--notebook-current-env``: uses the Python environment that launched canary.
2. ``--notebook-kernel-name NAME``: uses the named kernel.
3. The kernel name stored in the notebook's ``kernelspec`` metadata.
4. Fallback: ``python``.

Timeout configuration
---------------------

Timeouts can be set via CLI flags or via ``canary``'s timeout mechanism:

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Config path
     - Default
     - Description
   * - ``nb-cell`` (``--notebook-cell-timeout T``)
     - 2000 s
     - Maximum time per cell execution.
   * - ``nb-kernel-startup`` (``--notebook-kernel-startup-timeout T``)
     - 60 s
     - Maximum time to wait for the kernel to start.
