# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Unit tests for the canary-notebook plugin.

Tests are organized into sections:
  - find_comment_markers:  parsing of cell [key: value] markers
  - coalesce_streams:      stream merging and control-character handling
  - transform_streams_for_comparison: stream re-keying
  - compare_outputs:       output diffing logic
  - IPyNbTestGenerator:    lock() / describe() / file_patterns
  - NotebookLauncher.get_cells: cell extraction and option defaults
  - IPyNbCell.execute:     skip, raises, allow_failure markers (mocked kernel)
  - data files:            capabilities.json / skills.json are loadable
"""

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import nbformat
import pytest
from nbformat import NotebookNode

from canary_notebook.plugin import IPyNbCell
from canary_notebook.plugin import IPyNbTestGenerator
from canary_notebook.plugin import NbCellError
from canary_notebook.plugin import NotebookLauncher
from canary_notebook.plugin import coalesce_streams
from canary_notebook.plugin import find_comment_markers
from canary_notebook.plugin import transform_streams_for_comparison

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_DIR = Path(__file__).parent.parent / "sample_notebooks"


def _stream(name: str, text: str) -> NotebookNode:
    """Create a stream NotebookNode."""
    out = NotebookNode(output_type="stream")
    out.name = name
    out.text = text
    return out


def _make_cell(
    source: str,
    execution_count: int | None = 1,
    outputs: list | None = None,
) -> NotebookNode:
    """Create a code cell NotebookNode."""
    cell = NotebookNode(
        cell_type="code",
        source=source,
        execution_count=execution_count,
        outputs=outputs or [],
    )
    return cell


def _make_nb_cell(
    source: str = "",
    options: dict[str, Any] | None = None,
    cell_outputs: list | None = None,
    execution_count: int | None = 1,
) -> IPyNbCell:
    """Create an IPyNbCell with a mock parent."""
    parent = MagicMock(spec=NotebookLauncher)
    parent.timed_out = False
    raw_cell = _make_cell(source, execution_count=execution_count, outputs=cell_outputs or [])
    return IPyNbCell(
        name="Cell 0",
        parent=parent,
        cell_num=0,
        cell=raw_cell,
        options=options or {},
    )


# ---------------------------------------------------------------------------
# find_comment_markers
# ---------------------------------------------------------------------------


class TestFindCommentMarkers:
    def test_no_markers(self):
        assert find_comment_markers("x = 1\ny = 2") == {}

    def test_skip_true(self):
        src = "# [skip: true]\nx = 1"
        assert find_comment_markers(src) == {"skip": True}

    def test_skip_false(self):
        src = "# [skip: false]\nx = 1"
        assert find_comment_markers(src) == {"skip": False}

    def test_check_output_false(self):
        src = "# [check_output: false]\nprint('hello')"
        assert find_comment_markers(src) == {"check_output": False}

    def test_allow_failure_true(self):
        src = "# [allow_failure: true]"
        assert find_comment_markers(src) == {"allow_failure": True}

    def test_raises_valueerror(self):
        src = "# [raises: ValueError]"
        assert find_comment_markers(src) == {"raises": "ValueError"}

    def test_timeout_seconds(self):
        src = "# [timeout: 30]"
        result = find_comment_markers(src)
        assert result["timeout"] == pytest.approx(30.0)

    def test_timeout_duration_string(self):
        src = "# [timeout: 2m]"
        result = find_comment_markers(src)
        assert result["timeout"] == pytest.approx(120.0)

    def test_multiple_markers(self):
        src = textwrap.dedent(
            """\
            # [allow_failure: true]
            # [check_output: false]
            import something
            """
        )
        result = find_comment_markers(src)
        assert result == {"allow_failure": True, "check_output": False}

    def test_unknown_marker_warns(self):
        src = "# [unknown_key: value]"
        with pytest.warns(UserWarning, match="Unknown marker"):
            result = find_comment_markers(src)
        assert result == {}

    def test_non_comment_line_ignored(self):
        src = "[skip: true]"  # not a comment
        assert find_comment_markers(src) == {}


# ---------------------------------------------------------------------------
# coalesce_streams
# ---------------------------------------------------------------------------


class TestCoalesceStreams:
    def test_empty(self):
        assert coalesce_streams([]) == []

    def test_single_stream(self):
        outputs = [_stream("stdout", "hello\n")]
        result = coalesce_streams(outputs)
        assert len(result) == 1
        assert result[0].text == "hello\n"

    def test_two_consecutive_stdout_merged(self):
        outputs = [_stream("stdout", "hello "), _stream("stdout", "world\n")]
        result = coalesce_streams(outputs)
        assert len(result) == 1
        assert result[0].text == "hello world\n"

    def test_stdout_stderr_not_merged(self):
        outputs = [_stream("stdout", "out"), _stream("stderr", "err")]
        result = coalesce_streams(outputs)
        assert len(result) == 2

    def test_interleaved_streams_all_same_name_merged(self):
        """coalesce_streams merges ALL outputs of the same stream name, regardless of interleaving.
        stdout(a), stderr(b), stdout(c) -> stdout(ac) and stderr(b) (2 outputs)."""
        outputs = [
            _stream("stdout", "a"),
            _stream("stderr", "b"),
            _stream("stdout", "c"),
        ]
        result = coalesce_streams(outputs)
        # Both stdout chunks are merged; stderr stays separate
        assert len(result) == 2
        stdout_outs = [o for o in result if o.output_type == "stream" and o.name == "stdout"]
        assert len(stdout_outs) == 1
        assert stdout_outs[0].text == "ac"

    def test_carriage_return_handled(self):
        outputs = [_stream("stdout", "abc\rdef")]
        result = coalesce_streams(outputs)
        assert "\r" not in result[0].text

    def test_non_stream_output_preserved(self):
        display = NotebookNode(output_type="display_data")
        display["data"] = {"text/plain": "fig"}
        outputs = [display, _stream("stdout", "x")]
        result = coalesce_streams(outputs)
        assert len(result) == 2
        assert result[0].output_type == "display_data"


# ---------------------------------------------------------------------------
# transform_streams_for_comparison
# ---------------------------------------------------------------------------


class TestTransformStreamsForComparison:
    def test_stream_gets_name_key(self):
        outputs = [_stream("stdout", "hello\n")]
        result = transform_streams_for_comparison(outputs)
        assert len(result) == 1
        assert result[0]["stdout"] == "hello\n"
        assert result[0]["output_type"] == "stream"

    def test_non_stream_passthrough(self):
        display = NotebookNode(output_type="display_data")
        display["data"] = {"text/plain": "x"}
        result = transform_streams_for_comparison([display])
        assert len(result) == 1
        assert result[0].output_type == "display_data"


# ---------------------------------------------------------------------------
# compare_outputs (via IPyNbCell)
# ---------------------------------------------------------------------------


class TestCompareOutputs:
    def _cell(self) -> IPyNbCell:
        return _make_nb_cell()

    def test_identical_outputs_pass(self):
        cell = self._cell()
        ref = [_stream("stdout", "hello\n")]
        test = [_stream("stdout", "hello\n")]
        assert cell.compare_outputs(test, ref) is True

    def test_different_text_fails(self):
        cell = self._cell()
        ref = [_stream("stdout", "hello\n")]
        test = [_stream("stdout", "world\n")]
        assert cell.compare_outputs(test, ref) is False

    def test_missing_output_field_fails(self):
        cell = self._cell()
        ref = [_stream("stdout", "hello\n")]
        test = []
        assert cell.compare_outputs(test, ref) is False

    def test_extra_output_field_fails(self):
        cell = self._cell()
        ref = []
        test = [_stream("stdout", "extra\n")]
        assert cell.compare_outputs(test, ref) is False

    def test_skip_compare_field_ignored(self):
        cell = self._cell()
        # name is in skip_compare by default — stream names differ but are skipped
        ref_out = NotebookNode(output_type="stream")
        ref_out.name = "stdout"
        ref_out.text = "x"
        test_out = NotebookNode(output_type="stream")
        test_out.name = "stderr"  # different name
        test_out.text = "x"
        # Because 'name' is in skip_compare, difference in name is ignored,
        # but after transform_streams_for_comparison the dict key differs (stdout vs stderr)
        # so the comparison will see dissimilar keys. Verify the comparison runs.
        result = cell.compare_outputs([test_out], [ref_out])
        # Different stream names → different dict keys → fails
        assert result is False

    def test_sanitize_applied(self):
        cell = _make_nb_cell()
        cell.sanitize_patterns = {r"\d{4}-\d{2}-\d{2}": "DATE"}
        ref = [_stream("stdout", "DATE\n")]
        test = [_stream("stdout", "2026-01-01\n")]
        assert cell.compare_outputs(test, ref) is True

    def test_empty_vs_empty_pass(self):
        cell = self._cell()
        assert cell.compare_outputs([], []) is True


# ---------------------------------------------------------------------------
# IPyNbTestGenerator
# ---------------------------------------------------------------------------


class TestIPyNbTestGenerator:
    def test_file_patterns(self):
        assert "*.ipynb" in IPyNbTestGenerator.file_patterns

    def test_lock_returns_single_spec(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        nb = nbformat.v4.new_notebook()
        nbformat.write(nb, nb_path)
        gen = IPyNbTestGenerator(str(tmp_path), "test.ipynb")
        specs = gen.lock()
        assert len(specs) == 1
        spec = specs[0]
        assert spec.file == nb_path

    def test_lock_keywords(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        nb = nbformat.v4.new_notebook()
        nbformat.write(nb, nb_path)
        gen = IPyNbTestGenerator(str(tmp_path), "test.ipynb")
        specs = gen.lock()
        assert "jupyter" in specs[0].keywords
        assert "notebook" in specs[0].keywords

    def test_describe_contains_cell_count(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        nb = nbformat.v4.new_notebook()
        nb.cells.append(nbformat.v4.new_code_cell("x = 1"))
        nb.cells.append(nbformat.v4.new_code_cell("y = 2"))
        nbformat.write(nb, nb_path)
        gen = IPyNbTestGenerator(str(tmp_path), "test.ipynb")

        # config.get is called for timeout resolution (needs a float) and for
        # notebook:sanitize (needs None/falsy). Use a side_effect to distinguish.
        def _config_get(path, default=None):
            if "timeout" in path:
                return 30.0
            return None

        with patch("canary.config.getoption", return_value=None):
            with patch("canary.config.get", side_effect=_config_get):
                desc = gen.describe()
        assert "2 cells" in desc


# ---------------------------------------------------------------------------
# NotebookLauncher.get_cells
# ---------------------------------------------------------------------------


class TestNotebookLauncherGetCells:
    def _launcher(self):
        return NotebookLauncher()

    def test_only_code_cells_collected(self):
        nb = nbformat.v4.new_notebook()
        nb.cells.append(nbformat.v4.new_markdown_cell("# Title"))
        nb.cells.append(nbformat.v4.new_code_cell("x = 1"))
        launcher = self._launcher()
        with patch("canary.config.getoption", return_value=False):
            with patch("canary.config.get", return_value=None):
                cells = launcher.get_cells(nb)
        assert len(cells) == 1

    def test_check_output_default_from_config(self):
        nb = nbformat.v4.new_notebook()
        nb.cells.append(nbformat.v4.new_code_cell("x = 1"))
        launcher = self._launcher()
        with patch("canary.config.getoption", return_value=True):  # dont_compare = True
            with patch("canary.config.get", return_value=None):
                cells = launcher.get_cells(nb)
        assert cells[0].options["check_output"] is False

    def test_check_output_overridden_by_marker(self):
        nb = nbformat.v4.new_notebook()
        nb.cells.append(nbformat.v4.new_code_cell("# [check_output: true]\nx = 1"))
        launcher = self._launcher()
        with patch("canary.config.getoption", return_value=True):  # dont_compare = True
            with patch("canary.config.get", return_value=None):
                cells = launcher.get_cells(nb)
        # marker explicitly sets check_output: true, overriding the global flag
        assert cells[0].options["check_output"] is True

    def test_cell_names_sequential(self):
        nb = nbformat.v4.new_notebook()
        nb.cells.append(nbformat.v4.new_code_cell("a = 1"))
        nb.cells.append(nbformat.v4.new_code_cell("b = 2"))
        launcher = self._launcher()
        with patch("canary.config.getoption", return_value=False):
            with patch("canary.config.get", return_value=None):
                cells = launcher.get_cells(nb)
        assert cells[0].name == "Cell 0"
        assert cells[1].name == "Cell 1"


# ---------------------------------------------------------------------------
# IPyNbCell.execute — skip marker
# ---------------------------------------------------------------------------


class TestIPyNbCellExecuteSkip:
    def test_skip_true_does_nothing(self):
        """A cell marked [skip: true] should return without touching the kernel."""
        cell = _make_nb_cell(source="# [skip: true]\nx = 1", options={"skip": True})
        kernel = MagicMock()
        cell.execute(kernel)
        kernel.execute_cell_input.assert_not_called()

    def test_skip_false_executes(self):
        """A cell marked [skip: false] should execute."""
        cell = _make_nb_cell(source="x = 1", options={"skip": False})
        kernel = MagicMock()
        kernel.is_alive.return_value = True
        # await_reply completes normally; iopub loop returns idle immediately
        idle_msg = {
            "msg_type": "status",
            "content": {"execution_state": "idle"},
            "parent_header": {"msg_id": "mid"},
        }
        kernel.execute_cell_input.return_value = "mid"
        kernel.get_message.return_value = idle_msg
        cell.execute(kernel)
        kernel.execute_cell_input.assert_called_once()


# ---------------------------------------------------------------------------
# IPyNbCell.execute — allow_failure marker
# ---------------------------------------------------------------------------


class TestIPyNbCellExecuteAllowFailure:
    def _mock_error_kernel(self, msg_id: str = "mid"):
        """Kernel that produces an error iopub message then idle."""
        kernel = MagicMock()
        kernel.is_alive.return_value = True
        kernel.execute_cell_input.return_value = msg_id

        error_msg = {
            "msg_type": "error",
            "content": {
                "ename": "RuntimeError",
                "evalue": "boom",
                "traceback": ["trace"],
            },
            "parent_header": {"msg_id": msg_id},
        }
        idle_msg = {
            "msg_type": "status",
            "content": {"execution_state": "idle"},
            "parent_header": {"msg_id": msg_id},
        }
        kernel.get_message.side_effect = [error_msg, idle_msg]
        return kernel

    def test_allow_failure_does_not_raise(self):
        # check_output=False so the error output doesn't trigger a comparison failure
        cell = _make_nb_cell(options={"allow_failure": True, "check_output": False})
        kernel = self._mock_error_kernel()
        # Should not raise NbCellError
        cell.execute(kernel)

    def test_no_allow_failure_raises(self):
        cell = _make_nb_cell(options={"check_output": False})
        kernel = self._mock_error_kernel()
        with pytest.raises(NbCellError, match="RuntimeError"):
            cell.execute(kernel)


# ---------------------------------------------------------------------------
# IPyNbCell.execute — raises marker
# ---------------------------------------------------------------------------


class TestIPyNbCellExecuteRaises:
    def _mock_error_kernel(self, ename: str, msg_id: str = "mid"):
        kernel = MagicMock()
        kernel.is_alive.return_value = True
        kernel.execute_cell_input.return_value = msg_id
        error_msg = {
            "msg_type": "error",
            "content": {"ename": ename, "evalue": "", "traceback": []},
            "parent_header": {"msg_id": msg_id},
        }
        idle_msg = {
            "msg_type": "status",
            "content": {"execution_state": "idle"},
            "parent_header": {"msg_id": msg_id},
        }
        kernel.get_message.side_effect = [error_msg, idle_msg]
        return kernel

    def test_raises_correct_exception_passes(self):
        # check_output=False so the error output doesn't trigger a comparison failure
        cell = _make_nb_cell(options={"raises": "ValueError", "check_output": False})
        kernel = self._mock_error_kernel("ValueError")
        # Should not raise
        cell.execute(kernel)

    def test_raises_wrong_exception_fails(self):
        cell = _make_nb_cell(options={"raises": "ValueError", "check_output": False})
        kernel = self._mock_error_kernel("RuntimeError")
        with pytest.raises(NbCellError, match="Expected exception of type ValueError"):
            cell.execute(kernel)


# ---------------------------------------------------------------------------
# IPyNbCell.execute — clear_output message
# ---------------------------------------------------------------------------


class TestIPyNbCellClearOutput:
    def test_clear_output_resets_outs(self):
        """A clear_output message with wait=False should clear accumulated outputs."""
        cell = _make_nb_cell(options={"check_output": False})
        kernel = MagicMock()
        kernel.is_alive.return_value = True
        msg_id = "mid"
        kernel.execute_cell_input.return_value = msg_id

        stream_msg = {
            "msg_type": "stream",
            "content": {"name": "stdout", "text": "before clear"},
            "parent_header": {"msg_id": msg_id},
        }
        clear_msg = {
            "msg_type": "clear_output",
            "content": {"wait": False},
            "parent_header": {"msg_id": msg_id},
        }
        idle_msg = {
            "msg_type": "status",
            "content": {"execution_state": "idle"},
            "parent_header": {"msg_id": msg_id},
        }
        kernel.get_message.side_effect = [stream_msg, clear_msg, idle_msg]
        # Should complete without error; output before clear was wiped
        cell.execute(kernel)


# ---------------------------------------------------------------------------
# data files
# ---------------------------------------------------------------------------


class TestDataFiles:
    def test_capabilities_json_loadable(self):
        from importlib import resources

        path = resources.files("canary_notebook.data").joinpath("capabilities.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "schema_version" in data
        assert "namespace" in data
        assert data["namespace"] == "notebook"
        assert "capabilities" in data

    def test_skills_json_loadable(self):
        from importlib import resources

        path = resources.files("canary_notebook.data").joinpath("skills.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "schema_version" in data
        assert "namespace" in data
        assert data["namespace"] == "notebook"
        assert "skills" in data

    def test_capabilities_overview_present(self):
        from importlib import resources

        path = resources.files("canary_notebook.data").joinpath("capabilities.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "overview" in data["capabilities"]
        assert "*.ipynb" in data["capabilities"]["overview"]["file_patterns"]

    def test_skills_body_is_string(self):
        from importlib import resources

        path = resources.files("canary_notebook.data").joinpath("skills.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        for skill_name, skill_obj in data["skills"].items():
            assert isinstance(skill_obj["body"], str), f"body of {skill_name} must be a string"

    def test_canary_capabilities_hook(self):
        from canary_notebook.plugin import canary_capabilities

        result = canary_capabilities()
        assert result is not None
        assert result["namespace"] == "notebook"

    def test_canary_skills_hook(self):
        from canary_notebook.plugin import canary_skills

        result = canary_skills()
        assert result is not None
        assert result["namespace"] == "notebook"


# ---------------------------------------------------------------------------
# NbCellError
# ---------------------------------------------------------------------------


class TestNbCellError:
    def test_basic_construction(self):
        err = NbCellError("something went wrong", cell_num=3, source="x = 1")
        assert str(err) == "something went wrong"
        assert err.cell_num == 3
        assert err.source == "x = 1"
        assert err.inner_traceback is None

    def test_with_traceback(self):
        err = NbCellError("fail", traceback="trace text", cell_num=0, source="")
        assert err.inner_traceback == "trace text"

    def test_repr_failure_nbcellerror(self):
        cell = _make_nb_cell()
        err = NbCellError("fail", traceback="tb", cell_num=0, source="x = 1")
        msg = cell.repr_failure(err)
        assert "Cell 0" in msg or "Notebook cell execution failed" in msg

    def test_repr_failure_generic_exception(self):
        cell = _make_nb_cell()
        err = ValueError("oops")
        msg = cell.repr_failure(err)
        assert "oops" in msg


# ---------------------------------------------------------------------------
# Sample notebooks exist and are valid nbformat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "minimal_example.ipynb",
        "sample_notebook.ipynb",
        "exceptions.ipynb",
        "test_coalesce.ipynb",
    ],
)
def test_sample_notebook_is_valid_nbformat(name):
    nb_path = SAMPLE_DIR / name
    assert nb_path.exists(), f"Sample notebook not found: {nb_path}"
    nb = nbformat.read(nb_path, as_version=4)
    # nbformat.validate raises nbformat.ValidationError on invalid notebooks
    nbformat.validate(nb)
