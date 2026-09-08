# Canary Notebook: A Testing Extension for Jupyter Notebooks

`canary-notebook` is a [canary](https://canary-wm.readthedocs.io)
extension, inspired by [pytest-nbval](https://github.com/nteract/nbval), that
tests execution of Jupyter notebooks.

## How It Works

Each `.ipynb` file is treated as a **single test case**. All code cells are
executed sequentially in one live Jupyter kernel. If any cell raises an
unhandled exception the test fails, but execution of subsequent cells continues
so all errors are visible in one run.

Cell behavior is controlled by `# [key: value]` comment markers (see
[Cell markers](#cell-markers) below). Outputs can be compared against the
values stored in the notebook file, with optional regex-based sanitization for
non-deterministic content.

## Installation

```console
pip install canary-notebook
```

Or from source:

```console
git clone https://github.com/sandialabs/canary-notebook.git
cd canary-notebook
pip install [-e] .
```

## Usage

```console
canary run [options] path [path...]

Notebook options:
  --notebook-config FILE
  --notebook-current-env
  --notebook-kernel-name NAME
  --notebook-cell-timeout T
  --notebook-kernel-startup-timeout T
  --notebook-dont-compare-outputs
```

Run a single notebook:

```console
canary run path/to/notebook.ipynb
```

Use the current Python environment's kernel instead of the one stored in the notebook:

```console
canary run --notebook-current-env path/to/notebook.ipynb
```

Disable output comparison globally:

```console
canary run --notebook-dont-compare-outputs path/to/notebook.ipynb
```

Query the plugin's capability data (requires canary-wm):

```console
canary query -c ext.notebook.overview
canary query -c ext.notebook.cell_markers
canary query -c ext.notebook.cli_options
```

## Cell markers

Cell behavior can be overridden with `# [key: value]` comment markers at the
top of a cell:

| Marker | Effect |
|---|---|
| `# [skip: true]` | Do not execute this cell |
| `# [check_output: false]` | Execute but do not compare outputs |
| `# [check_output: true]` | Force output comparison (overrides global flag) |
| `# [allow_failure: true]` | Cell errors do not fail the test |
| `# [raises: ExceptionType]` | Cell must raise the named exception |
| `# [timeout: T]` | Per-cell timeout (seconds or duration string e.g. `5m`) |

Example:

```python
# [raises: ValueError]
raise ValueError("expected error")
```

## Output comparison

By default, cell outputs are compared against outputs stored in the notebook
file. The following fields are always excluded: `metadata`, `traceback`,
`execution_count`, widget view IDs, `image/png`, `image/jpeg`.

To sanitize non-deterministic output, pass a YAML config file:

```yaml
# sanitize.yaml
notebook:
  sanitize:
    - regex: '\d{4}-\d{2}-\d{2}'
      replace: 'DATE'
    - regex: '0x[0-9a-fA-F]+'
      replace: '0xADDR'
```

```console
canary run --notebook-config sanitize.yaml path/to/notebook.ipynb
```

## Acknowledgments

`canary-notebook` is inspired by and borrows kernel infrastructure from
[`pytest-nbval`](https://github.com/nteract/nbval).

## License

`canary-notebook` is distributed under the terms of the MIT license. See
[LICENSE](https://github.com/sandialabs/canary-notebook/blob/main/LICENSE) and
[COPYRIGHT](https://github.com/sandialabs/canary-notebook/blob/main/COPYRIGHT).

SPDX-License-Identifier: MIT

SCR#:3170.0
