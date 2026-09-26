# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""canary ipython notebook plugin"""

import argparse
import datetime
import hashlib
import io
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path
from queue import Empty
from typing import Any
from typing import ClassVar
from typing import Sequence

import canary

# for reading notebook files
import nbformat
import yaml
from nbformat import NotebookNode

from .kernel import CURRENT_ENV_KERNEL_NAME
from .kernel import RunningKernel

logger = canary.get_logger(__name__)
colorize = canary.color.colorize

DEFAULT_KERNEL_STARTUP_TIMEOUT = 60
# 2000 s is intentionally large — "practically unlimited"; individual cells can
# override via the [timeout: T] marker or --notebook-cell-timeout.
DEFAULT_CELL_TIMEOUT = 2000


@canary.hookimpl
def canary_addoption(parser: canary.Parser) -> None:
    def addoption(name, **kwargs):
        parser.add_argument(
            f"--notebook-{name}",
            dest=f"canary_notebook_{name.replace('-', '_')}",
            group="canary notebook",
            command="run",
            **kwargs,
        )

    addoption(
        "config",
        metavar="FILE",
        help="YAML config file with regex expressions to sanitize the outputs.",
    )

    addoption(
        "current-env",
        action="store_true",
        help="Use a python kernel in the same environment that canary was launched from. "
        "Without this flag, the kernel stored in the notebook is used by default.",
    )

    addoption(
        "kernel-name",
        metavar="NAME",
        action="store",
        default=None,
        help="Use the named kernel. If a kernel is not named, the kernel stored in the "
        "notebook is used by default.",
    )

    addoption(
        "cell-timeout",
        metavar="T",
        action=TimeoutFlag,
        help="Timeout for cell execution, in seconds (alias for --timeout nb-cell:T)",
    )

    addoption(
        "kernel-startup-timeout",
        metavar="T",
        action=TimeoutFlag,
        help="Timeout for kernel startup, in seconds - alias for --timeout nb-kernel-startup=T",
    )

    addoption(
        "dont-compare-outputs",
        action="store_true",
        help="Don't compare notebook cell outputs",
    )


class TimeoutFlag(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        timeouts = getattr(namespace, "timeouts", None) or {}
        assert option_string is not None
        type = option_string.replace("--notebook-", "nb-").rstrip("-timeout")
        assert isinstance(values, str)
        timeouts[type] = float(values)
        setattr(namespace, "timeouts", timeouts)


config_file_schema = canary.schema.Schema(
    {"notebook": {canary.schema.Optional("sanitize"): [{"regex": str, "replace": str}]}}
)


@canary.hookimpl
def canary_configure(config: canary.Config):
    if config.getoption("canary_notebook_kernel_name") and config.getoption(
        "canary_notebook_current_env"
    ):
        raise ValueError(
            "options '--notebook-current-env' and '--notebook-kernel-name' are mutually exclusive."
        )
    if file := config.getoption("canary_notebook_config"):
        if not os.path.isabs(file):
            file = os.path.join(config.invocation_dir, file)
        if not os.path.exists(file):
            f = config.getoption("canary_notebook_config")
            raise FileNotFoundError(f)
        with open(file, "r") as fh:
            data = yaml.safe_load(fh)
        data = config_file_schema.validate(data)
        config.add_section(name="notebook", schema=config_file_schema)
        config.set("notebook", data)


@canary.hookimpl
def canary_capabilities() -> dict[str, Any] | None:
    return canary.load_query_data("canary_notebook.data", "capabilities.json")


@canary.hookimpl
def canary_skills() -> dict[str, Any] | None:
    return canary.load_query_data("canary_notebook.data", "skills.json")


@canary.hookimpl
def canary_runtest_launcher(case: "canary.Job") -> "NotebookLauncher | None":
    if case.file.suffix == ".ipynb":
        return NotebookLauncher()
    return None


@canary.hookimpl
def canary_collectstart(collector: canary.Collector) -> None:
    collector.add_generator(IPyNbTestGenerator)


@canary.hookimpl
def canary_collect_modifyitems(collector: canary.Collector) -> None:
    checkpoint_files: list[tuple[str, str]] = []
    for root, path in collector.iter_files():
        file = Path(os.path.join(root, path))
        if ".ipynb_checkpoints" in file.parts:
            checkpoint_files.append((root, path))
    for root, path in checkpoint_files:
        collector.remove_file(root, path)


def find_comment_markers(cellsource: str) -> dict[str, Any]:
    """Look through the cell source for comments which affect notebook's behaviour"""
    known_comment_markers: tuple[str, ...] = (
        "skip",
        "allow_failure",
        "check_output",
        "timeout",
        "raises",
    )
    markers: dict[str, Any] = {}
    for line in cellsource.splitlines():
        line = line.strip()
        if line.startswith("#"):
            comment = line.lstrip("#").strip()
            for type, value in re.findall(r"\[(\w+):\s*(.*?)\]", comment):
                if type not in known_comment_markers:
                    warnings.warn(f"Unknown marker: [{type}: {value}]")
                    continue
                value = yaml.safe_load(value)
                if type in ("timeout",):
                    value = canary.time.time_in_seconds(value)
                markers[type] = value
    return markers


class IPyNbTestGenerator(canary.AbstractTestGenerator):
    """
    This class represents a canary test case generator.
    A generator is associated with an ipynb file and collects the cells
    in the notebook for testing.
    """

    file_patterns: ClassVar[tuple[str, ...]] = ("*.ipynb",)

    def __init__(self, root: str, path: str | None = None) -> None:
        super().__init__(root, path=path)

    def lock(self, on_options: list[str] | None = None) -> list[canary.ResolvedSpec]:
        kwargs: dict[str, Any] = {}
        kwargs["file_root"] = Path(self.root)
        kwargs["file_path"] = Path(self.path)
        kwargs["keywords"] = self.keywords()
        return [canary.ResolvedSpec(**kwargs)]

    def keywords(self) -> list[str]:
        return ["jupyter", "notebook"]

    def describe(self, on_options: list[str] | None = None) -> str:
        file = io.StringIO()
        file.write(f"--- {self.name} ------------\n")
        file.write(f"File: {self.file}\n")
        file.write(f"Keywords: {', '.join(self.keywords())}\n")
        spec = self.lock(on_options=on_options)[0]
        nb = nbformat.read(spec.file, as_version=4)
        l = NotebookLauncher()
        cells = l.get_cells(nb)
        n = len(cells)
        file.write(f"1 test case with {n} cell{'s' if n > 1 else ''}")
        return file.getvalue()


class NotebookLauncher(canary.Launcher):
    def __init__(self) -> None:
        self.timed_out: bool = False
        self.skip_compare: list[str] = [
            "metadata",
            "traceback",
            #'text/latex',
            "prompt_number",
            "output_type",
            "name",
            "execution_count",
            "application/vnd.jupyter.widget-view+json",  # Model IDs are random
            "image/png",
            "image/jpeg",
        ]

    @staticmethod
    def read_sanitize_patterns() -> dict[str, str]:
        patterns: dict[str, str] = {}
        if items := canary.config.get("notebook:sanitize"):
            patterns.update({item["regex"]: item["replace"] for item in items})
        return patterns

    @staticmethod
    def start_kernel(file: Path, kernelspec: dict[str, Any]) -> RunningKernel:
        name: str
        if canary.config.getoption("canary_notebook_current_env"):
            name = CURRENT_ENV_KERNEL_NAME
        elif canary.config.getoption("canary_notebook_kernel_name"):
            name = canary.config.getoption("canary_notebook_kernel_name")
        else:
            name = kernelspec.get("name", "python")
        timeout: float = DEFAULT_KERNEL_STARTUP_TIMEOUT
        if user_defined_timeout := canary.config.get("config:run:timeout:nb-kernel-startup"):
            timeout = user_defined_timeout
        kernel = RunningKernel(name, cwd=os.path.dirname(file), startup_timeout=timeout)
        return kernel

    def get_cells(self, nb: nbformat.NotebookNode) -> list["IPyNbCell"]:
        # Read through the specified notebooks and load the data
        # (which is in json format)
        cells: list[IPyNbCell] = []
        compare_outputs = not canary.config.getoption("canary_notebook_dont_compare_outputs")
        sanitize_patterns = self.read_sanitize_patterns()
        cell_num = 0
        for cell in nb.cells:
            # Skip the cells that have text, headings or related stuff
            # Only test code cells
            if cell.cell_type == "code":
                # The cell may contain a comment indicating that its output
                # should be checked or ignored. If it doesn't, use the default
                # behaviour.
                options = find_comment_markers(cell.source)
                options.setdefault("check_output", compare_outputs)
                name = "Cell " + str(cell_num)
                cell = IPyNbCell(
                    name,
                    self,
                    cell_num=cell_num,
                    cell=cell,
                    options=options,
                    sanitize_patterns=sanitize_patterns,
                    skip_compare=self.skip_compare,
                )
                cells.append(cell)
                cell_num += 1
        return cells

    def run(self, job: "canary.Job") -> int:
        logger.debug(f"Starting {job.display_name()} on pid {os.getpid()}")
        timeout = canary.config.get("config:run:timeout:nb-cell")
        nb = nbformat.read(job.file, as_version=4)
        kernel = self.start_kernel(job.file, nb.metadata.get("kernelspec", {}))
        with warnings.catch_warnings(record=True) as ws:
            cells = self.get_cells(nb)
        prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
        assert job.stdout is not None
        with job.workspace.openfile(job.stdout, "a") as fp:
            fp.write(f"[{prefix}] Running {job.display_name()}\n")
            fp.write(f"[{prefix}] Working directory: {job.workspace.dir}\n")
            fp.write(f"[{prefix}] Execution directory: {job.file.parent}\n")
            fp.write(f"[{prefix}] {len(cells)} cells to execute\n")
        for w in ws:
            logger.warning(str(w.message))
        errors: int = 0
        try:
            with canary.filesystem.working_dir(job.file.parent):
                for cell in cells:
                    try:
                        cell.execute(kernel, timeout=timeout)
                    except Exception as e:
                        msg = cell.repr_failure(e)
                        f = job.stdout or job.stderr
                        assert f is not None
                        with job.workspace.openfile(f, "a") as fp:
                            fp.write(msg + "\n")
                        errors += 1
        finally:
            if kernel.is_alive():
                kernel.stop()
        success = len(cells) - errors
        with job.workspace.openfile(job.stdout, "a") as fp:
            fp.write(f"[{prefix}] {len(cells)} total cells, {success} cells pass, {errors} fail\n")
        logger.debug(f"Finished {job.display_name()}")
        return 1 if errors else 0


class IPyNbCell:
    def __init__(
        self,
        name: str,
        parent: NotebookLauncher,
        cell_num: int,
        cell,
        options: dict[str, Any],
        sanitize_patterns: dict[str, str] | None = None,
        skip_compare: list[str] | None = None,
        **kwds,
    ) -> None:
        self.name = name
        self.parent = parent
        self.cell_num = cell_num
        self.cell = cell
        self.options = options
        self.output_timeout = 5
        self.sanitize_patterns: dict[str, str] = sanitize_patterns or {}
        self.skip_compare = skip_compare or []

    def repr_failure(self, exc: BaseException) -> str:
        """called when self.runtest() raises an exception."""
        if isinstance(exc, NbCellError):
            width: int = 88
            msg = io.StringIO()
            msg.write("=" * width)
            msg.write("\n[bold red]Notebook cell execution failed[/]\n")
            msg.write(
                "[bold blue]Cell %d: %s\nInput:[/]\n%s\n" % (exc.cell_num, str(exc), exc.source)
            )
            if exc.inner_traceback:
                msg.write("[bold blue]Traceback[/]:%s\n" % exc.inner_traceback)
            return colorize(msg.getvalue())
        else:
            return "canary-notebook plugin exception: %s" % str(exc)

    def compare_outputs(self, test, ref, skip_compare=None):
        # Use stored skips unless passed a specific value
        skip_compare = skip_compare or self.skip_compare

        test = transform_streams_for_comparison(test)
        ref = transform_streams_for_comparison(ref)

        # We reformat outputs into a dictionaries where
        # key:
        #   - all keys on output except 'data' and those in skip_compare
        #   - all keys on 'data' except those in skip_compare, i.e. data is flattened
        # value:
        #   - list of all corresponding values for that key, i.e. for all outputs
        #
        # This format allows to disregard the relative order of dissimilar
        # output keys, while still caring about the order of those that share
        # a key.
        testing_outs = defaultdict(list)
        reference_outs = defaultdict(list)

        for reference in ref:
            for key in reference:
                if key in skip_compare:
                    continue
                # Flatten out MIME types from data of display_data and execute_result
                if key == "data":
                    for data_key in reference[key]:
                        # Filter the keys in the SUB-dictionary again:
                        if data_key not in skip_compare:
                            sanitized = self.sanitize(reference[key][data_key])
                            reference_outs[data_key].append(sanitized)

                # Otherwise, just create a normal dictionary entry from
                # one of the keys of the dictionary
                else:
                    # Create the dictionary entries on the fly, from the
                    # existing ones to be compared
                    reference_outs[key].append(self.sanitize(reference[key]))

        # the same for the testing outputs (the cells that are being executed)
        for testing in test:
            for key in testing:
                if key in skip_compare:
                    continue
                if key == "data":
                    for data_key in testing[key]:
                        if data_key not in skip_compare:
                            testing_outs[data_key].append(self.sanitize(testing[key][data_key]))
                else:
                    testing_outs[key].append(self.sanitize(testing[key]))

        # The traceback from the comparison will be stored here.
        self.comparison_traceback = []

        ref_keys = set(reference_outs)
        test_keys = set(testing_outs)

        if ref_keys - test_keys:
            msg = "[bold red]Missing output fields from running code: %s[/]" % (
                ref_keys - test_keys
            )
            self.comparison_traceback.append(colorize(msg))
            return False
        elif test_keys - ref_keys:
            msg = "[bold red]Unexpected output fields from running code: %s[/]" % (
                test_keys - ref_keys
            )
            self.comparison_traceback.append(colorize(msg))
            return False

        # If we've got to here, the two dicts must have the same set of keys

        for key in reference_outs:
            # Get output values for dictionary entries.
            # We use str() to be sure that the unicode key strings from the
            # reference are also read from the testing dictionary:
            test_values = testing_outs[str(key)]
            ref_values = reference_outs[key]
            if len(test_values) != len(ref_values):
                # The number of outputs for a specific MIME type differs
                msg = io.StringIO()
                msg.write('[bold blue]dissimilar number of outputs for key "%s"[/]' % key)
                msg.write("[bold red]<<<<<<<<<<<< Reference outputs from ipynb file:[/]")
                self.comparison_traceback.append(colorize(msg.getvalue()))
                for val in ref_values:
                    self.comparison_traceback.append(_trim_base64(val))
                self.comparison_traceback.append(
                    colorize(
                        "[bold red]============ disagrees with newly computed (test) output:[/]"
                    )
                )
                for val in test_values:
                    self.comparison_traceback.append(_trim_base64(val))
                self.comparison_traceback.append(colorize("[bold red]>>>>>>>>>>>>[/]"))
                return False

            for ref_out, test_out in zip(ref_values, test_values):
                # Compare the individual values
                if ref_out != test_out:
                    self.format_output_compare(key, ref_out, test_out)
                    return False
        return True

    def format_output_compare(self, key, left, right):
        """Format an output for printing"""
        if isinstance(left, str):
            left = _trim_base64(left)
        if isinstance(right, str):
            right = _trim_base64(right)

        self.comparison_traceback.append(colorize("[bold blue] mismatch '%s'[/]" % key))
        # Fallback repr:
        self.comparison_traceback.append(
            colorize("[bold red]  <<<<<<<<<<<< Reference output from ipynb file:[/]")
        )
        self.comparison_traceback.append(_indent(left))
        self.comparison_traceback.append(
            colorize("[bold red]  ============ disagrees with newly computed (test) output:[/]")
        )
        self.comparison_traceback.append(_indent(right))
        self.comparison_traceback.append(colorize("[bold red]  >>>>>>>>>>>>[/]"))

    """ *****************************************************
        ***************************************************** """

    def execute(self, kernel: RunningKernel, timeout: float | None = None) -> None:
        """
        Run test is called by canary for each of these nodes that are
        collected i.e. a notebook cell. Runs all the cell tests in one
        kernel without restarting.  It is very common for ipython
        notebooks to run through assuming a single kernel.  The cells
        are tested that they execute without errors and that the
        output matches the output stored in the notebook.

        """
        if self.options.get("skip", False):
            return

        assert kernel is not None
        if not kernel.is_alive():
            raise RuntimeError("Kernel dead on test start")

        # Execute the code in the current cell in the kernel. Returns the
        # message id of the corresponding response from iopub.
        msg_id = kernel.execute_cell_input(self.cell.source, allow_stdin=False)

        # Timeout for the cell execution
        # after code is sent for execution, the kernel sends a message on
        # the shell channel. Timeout if no message received.
        timed_out_this_run = False

        if timeout is None:
            if t := self.options.get("timeout"):
                timeout = t
            else:
                timeout = DEFAULT_CELL_TIMEOUT
        assert timeout is not None

        # Poll the shell channel to get a message
        try:
            kernel.await_reply(msg_id, timeout=timeout)
        except Empty:  # Timeout reached
            # Try to interrupt kernel, as this will give us traceback:
            kernel.interrupt()
            self.parent.timed_out = True
            timed_out_this_run = True

        # This list stores the output information for the entire cell
        outs: list[Any] = []

        # Now get the outputs from the iopub channel
        while True:
            # The iopub channel broadcasts a range of messages. We keep reading
            # them until we find the message containing the side-effects of our
            # code execution.
            try:
                # Get a message from the kernel iopub channel
                msg = kernel.get_message("iopub", timeout=self.output_timeout)

            except Empty:
                # This is not working: ! The code will not be checked
                # if the time is out (when the cell stops to be executed?)
                # Halt kernel here!
                kernel.stop()
                if timed_out_this_run:
                    raise NbCellError(
                        "Timeout of %g seconds exceeded while executing cell."
                        " Failed to interrupt kernel in %d seconds, so "
                        "failing without traceback." % (timeout, self.output_timeout),
                        cell_num=self.cell_num,
                        source=self.cell.source,
                    )
                else:
                    self.parent.timed_out = True
                    raise NbCellError(
                        "Timeout of %d seconds exceeded waiting for output." % self.output_timeout,
                        cell_num=self.cell_num,
                        source=self.cell.source,
                    )

            # now we must handle the message by checking the type and reply
            # info and we store the output of the cell in a notebook node object
            msg_type = msg["msg_type"]
            reply = msg["content"]
            out = NotebookNode(output_type=msg_type)

            # Is the iopub message related to this cell execution?
            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            # When the kernel starts to execute code, it will enter the 'busy'
            # state and when it finishes, it will enter the 'idle' state.
            # The kernel will publish state 'starting' exactly
            # once at process startup.
            if msg_type == "status":
                if reply["execution_state"] == "idle":
                    break
                else:
                    continue

            # execute_input: To let all frontends know what code is
            # being executed at any given time, these messages contain a
            # re-broadcast of the code portion of an execute_request,
            # along with the execution_count.
            elif msg_type == "execute_input":
                continue

            # com? execute reply?
            elif msg_type.startswith("comm"):
                continue
            elif msg_type == "execute_reply":
                continue

            # This message type is used to clear the output that is
            # visible on the frontend.  Reset accumulated outputs for
            # the current cell so comparisons are not contaminated with
            # pre-clear output.
            elif msg_type == "clear_output":
                if not reply.get("wait", False):
                    outs = []
                continue

            # 'execute_result' is equivalent to a display_data message.
            # The object being displayed is passed to the display
            # hook, i.e. the *result* of the execution.
            # The only difference is that 'execute_result' has an
            # 'execution_count' number which does not seems useful
            # (we will filter it in the sanitize function)
            #
            # When the reply is display_data or execute_result,
            # the dictionary contains
            # a 'data' sub-dictionary with the 'text' AND the 'image/png'
            # picture (in hexadecimal). There is also a 'metadata' entry
            # but currently is not of much use, sometimes there is information
            # as height and width of the image (CHECK the documentation)
            # Thus we iterate through the keys (mimes) 'data' sub-dictionary
            # to obtain the 'text' and 'image/png' information
            elif msg_type in ("display_data", "execute_result"):
                out["metadata"] = reply["metadata"]
                out["data"] = reply["data"]
                outs.append(out)

                if msg_type == "execute_result":
                    out.execution_count = reply["execution_count"]

            # if the message is a stream then we store the output
            elif msg_type == "stream":
                out.name = reply["name"]
                out.text = reply["text"]
                outs.append(out)

            # if the message type is an error then an error has occurred during
            # cell execution. Therefore raise a cell error and pass the
            # traceback information.
            elif msg_type == "error":
                # Store error in output first
                out["ename"] = reply["ename"]
                out["evalue"] = reply["evalue"]
                out["traceback"] = reply["traceback"]
                outs.append(out)
                if self.options.get("allow_failure", False):
                    continue
                elif raises := self.options.get("raises"):
                    if out["ename"] != raises:
                        raise NbCellError(
                            f"Expected exception of type {raises}, got {out['ename']}",
                            traceback="\n" + "\n".join(reply["traceback"]),
                            cell_num=self.cell_num,
                            source=self.cell.source,
                        )
                else:
                    # Ensure we flush iopub before raising error
                    try:
                        kernel.await_idle(msg_id, self.output_timeout)
                    except Empty:
                        kernel.stop()
                        raise RuntimeError("Timed out waiting for idle kernel!")
                    traceback = "\n" + "\n".join(reply["traceback"])
                    if out["ename"] == "KeyboardInterrupt" and self.parent.timed_out:
                        msg = f"Timeout of {timeout:g} seconds exceeded executing cell"
                    else:
                        msg = f"Cell execution raised {out['ename']!r}"
                    raise NbCellError(
                        msg,
                        traceback=traceback,
                        cell_num=self.cell_num,
                        source=self.cell.source,
                    )

            # any other message type is not expected
            else:
                logger.debug("unhandled iopub msg: %s", msg_type)

        outs[:] = coalesce_streams(outs)

        # Cells where the reference is not run, will not check outputs:
        unrun = self.cell.execution_count is None
        if unrun and self.cell.outputs:
            raise NbCellError(
                "Unrun reference cell has outputs", cell_num=self.cell_num, source=self.cell.source
            )

        # Compare if the outputs have the same number of lines
        # and throw an error if it fails
        failed = False
        if self.options.get("check_output", True) and not unrun:
            if not self.compare_outputs(outs, coalesce_streams(self.cell.outputs)):
                failed = True

        # If the comparison failed then we raise an exception.
        if failed:
            # The traceback containing the difference in the outputs is
            # stored in the variable comparison_traceback
            raise NbCellError(
                "Cell outputs differ",
                # Here we must put the traceback output:
                "\n".join(self.comparison_traceback),
                cell_num=self.cell_num,
                source=self.cell.source,
            )

    def sanitize(self, s):
        """sanitize a string for comparison."""
        if not isinstance(s, str):
            return s

        """
        re.sub matches a regex and replaces it with another.
        The regex replacements are taken from a file if the option
        is passed when py.test is called. Otherwise, the strings
        are not processed
        """
        for regex, replace in self.sanitize_patterns.items():
            s = re.sub(regex, replace, s)
        return s


carriagereturn_pat = re.compile(r"^.*\r(?=[^\n])", re.MULTILINE)
backspace_pat = re.compile(r"[^\n]\b")


def coalesce_streams(outputs):
    """
    Merge all stream outputs with shared names into single streams
    to ensure deterministic outputs.

    Parameters
    ----------
    outputs : iterable of NotebookNodes
        Outputs being processed
    """
    if not outputs:
        return outputs

    new_outputs = []
    streams = {}
    for output in outputs:
        if output.output_type == "stream":
            if output.name in streams:
                streams[output.name].text += output.text
            else:
                new_outputs.append(output)
                streams[output.name] = output
        else:
            new_outputs.append(output)

    # process \r and \b characters
    for output in streams.values():
        old = output.text
        while len(output.text) < len(old):
            old = output.text
            # Cancel out anything-but-newline followed by backspace
            output.text = backspace_pat.sub("", output.text)
        # Replace all carriage returns not followed by newline
        output.text = carriagereturn_pat.sub("", output.text)

    return new_outputs


def transform_streams_for_comparison(outputs):
    """Makes failure output for streams better by having key be the stream name"""
    new_outputs = []
    for output in outputs:
        if output.output_type == "stream":
            # Transform output
            new_outputs.append(
                {
                    "output_type": "stream",
                    output.name: output.text,
                }
            )
        else:
            new_outputs.append(output)
    return new_outputs


def hash_string(s: str) -> str:
    return hashlib.md5(s.encode("utf8")).hexdigest()


_base64 = re.compile(
    r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$", re.MULTILINE | re.UNICODE
)


def _trim_base64(s):
    """Trim and hash base64 strings"""
    if len(s) > 64 and _base64.match(s.replace("\n", "")):
        h = hash_string(s)
        s = "%s...<snip base64, md5=%s...>" % (s[:8], h[:16])
    return s


def _indent(s, indent="  "):
    """Intent each line with indent"""
    if isinstance(s, str):
        return "\n".join(("%s%s" % (indent, line) for line in s.splitlines()))
    return s


class NbCellError(Exception):
    """custom exception for error reporting."""

    def __init__(self, msg, traceback=None, cell_num=None, source=None):
        self.cell_num = cell_num
        self.source = source
        self.inner_traceback = traceback
        super().__init__(msg)
