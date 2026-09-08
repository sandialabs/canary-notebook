.. Copyright NTESS. See COPYRIGHT file for details.

   SPDX-License-Identifier: MIT

.. _notebook-install:

Installation
============

Requirements
------------

* Python 3.10+
* canary-wm >= 25.8.28
* ``jupyter_client``, ``nbformat``, ``ipykernel``, ``pyyaml`` (installed automatically)

Install
-------

.. code-block:: console

   pip install canary-notebook

Verify that the plugin is registered:

.. code-block:: console

   canary query -c ext.notebook.overview
