.. Copyright NTESS. See COPYRIGHT file for details.

   SPDX-License-Identifier: MIT

.. _notebook-comparison:

Output comparison
=================

By default, ``canary-notebook`` compares each cell's outputs against the
outputs stored in the notebook file.  The comparison is performed after applying
stream coalescing (consecutive outputs for the same stream name are merged) and
any configured sanitize patterns.

Always-excluded fields
----------------------

The following output fields are excluded from comparison regardless of cell
markers:

* ``metadata``
* ``traceback``
* ``prompt_number``
* ``output_type``
* ``name``
* ``execution_count``
* ``application/vnd.jupyter.widget-view+json`` (widget model IDs are random)
* ``image/png`` and ``image/jpeg`` (binary blobs)

Disabling comparison
--------------------

* Per cell: add ``# [check_output: false]`` to the cell source.
* Globally: pass ``--notebook-dont-compare-outputs`` to ``canary run``.

A cell marked ``# [check_output: false]`` is still executed; only the comparison
step is skipped.  Conversely, ``# [check_output: true]`` re-enables comparison
for a single cell even when the global flag is set.


Stream coalescing
-----------------

All outputs for the same stream name (``stdout`` or ``stderr``) are merged into
a single output before comparison.  This makes comparison deterministic when
the notebook's kernel interleaves stream writes differently from the reference
run.

Control characters ``\\r`` (carriage return) and ``\\b`` (backspace) are also
processed:  a ``\\r`` not followed by ``\\n`` overwrites the preceding line, and
``\\b`` cancels the preceding non-newline character.
