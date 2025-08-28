# Canary Notebook: A Testing Extension for Jupyter Notebooks

`canary-notebook` is a [canary](https://canary-wm.readthedocs.io/en/production/) extension, inspired by the [pytest-nbval](https://github.com/nteract/nbval), that tests execution of Jupyter notebooks.

## How It Works

The `canary-notebook` extension finds and executes Jupyter notebooks.  Each notebook is treated as a single test. When executed, cells in the notebook are run in sequential order.  If a cell fails during execution, the overall test is marked as failed; however, the execution of subsequent cells continues.

`canary-notebook` uses `nbval`'s Jupyter kernel interface which interacts with the IPython Kernel through both a `shell` and an `iopub` socket. The `shell` is responsible for executing the cells in the notebook by sending requests to the Kernel, while the `iopub` socket facilitates the retrieval of output messages. The messages received from the Kernel are organized into dictionaries containing various information, such as execution timestamps, cell data types, cell types, Kernel status, and username, among other details.

## Installation

To install Canary Notebook, you can use pip:

```console
pip install canary-notebook
```

to install the latest version:

```console
git clone git@cee-gitlab.sandia.gov:sandialabs/canary-notebook
cd canary-notebook
pip install [-e] .
```

## Usage

```console
canary run [options]
  [--notebook-config NOTEBOOK_CONFIG]
  [--notebook-current-env | --notebook-kernel-name NOTEBOOK_KERNEL_NAME]
  [--notebook-cell-timeout NOTEBOOK_CELL_TIMEOUT]
  [--notebook-kernel-startup-timeout NOTEBOOK_KERNEL_STARTUP_TIMEOUT]
  path [path...]
```

## Acknowledgments

`canary-notebook` is inspired by and borrows components from the [`pytest-nbval`](https://github.com/nteract/nbval) pytest extension.


## License

Canary is distributed under the terms of the MIT license, see [LICENSE](https://github.com/sandialabs/canary-notebook/blob/main/LICENSE) and [COPYRIGHT](https://github.com/sandialabs/canary-notebook/blob/main/COPYRIGHT).

SPDX-License-Identifier: MIT

SCR#:3170.0
