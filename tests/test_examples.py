"""Smoke tests for the examples.py fixture: sample data generation and its RecordGenerator."""

import json
from pathlib import Path

import polars as pl
import pytest

from trendify.examples import example_record_generator, make_example_data
from trendify.formats.format2d import Format2D
from trendify.formats.table import TableEntry
from trendify.plotting.axline import AxLine
from trendify.plotting.histogram import HistogramEntry
from trendify.plotting.point import Point2D
from trendify.plotting.trace import Trace2D


class TestMakeExampleData:
    def test_writes_one_folder_per_run_with_raw_data(self, tmp_path: Path) -> None:
        make_example_data(tmp_path, n_folders=3)

        model_dirs = sorted((tmp_path / "models").glob("*/"))
        assert [p.name for p in model_dirs] == ["0", "1", "2"]
        for model_dir in model_dirs:
            assert (model_dir / "results.csv").exists()
            assert (model_dir / "stdin.csv").exists()

    def test_is_deterministic_per_run_index(self, tmp_path: Path) -> None:
        # Each run seeds its RNG from its own index, so regenerating must reproduce the
        # same raw data rather than drawing fresh random values every time.
        make_example_data(tmp_path / "a", n_folders=1)
        make_example_data(tmp_path / "b", n_folders=1)
        first = (tmp_path / "a" / "models" / "0" / "results.csv").read_text()
        second = (tmp_path / "b" / "models" / "0" / "results.csv").read_text()
        assert first == second

    def test_n_cols_adds_filler_columns_without_changing_the_waveforms(
        self, tmp_path: Path
    ) -> None:
        make_example_data(tmp_path / "narrow", n_folders=1)
        make_example_data(tmp_path / "wide", n_folders=1, n_cols=10)

        narrow = pl.read_csv(tmp_path / "narrow" / "models" / "0" / "results.csv")
        wide = pl.read_csv(tmp_path / "wide" / "models" / "0" / "results.csv")

        assert narrow.columns == ["time", "wave_1", "wave_2", "wave_3"]
        assert wide.columns == ["time", *(f"wave_{i}" for i in range(1, 11))]
        assert wide.select(narrow.columns).equals(narrow)

    def test_n_cols_below_the_three_waveforms_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="n_cols"):
            make_example_data(tmp_path, n_folders=1, n_cols=2)


class TestExampleRecordGenerator:
    def test_ignores_filler_columns(self, tmp_path: Path) -> None:
        make_example_data(tmp_path / "narrow", n_folders=1)
        make_example_data(tmp_path / "wide", n_folders=1, n_cols=10)

        narrow = example_record_generator(tmp_path / "narrow" / "models" / "0")
        wide = example_record_generator(tmp_path / "wide" / "models" / "0")

        # Polars parses a wider file in more chunks and sums per chunk, so aggregates like a
        # mean can differ in the last bit between the two files even though the data is
        # identical. Rounding floats keeps that summation-order noise out of the comparison.
        def normalized(records) -> list[object]:
            return [
                json.loads(
                    r.model_dump_json(), parse_float=lambda s: round(float(s), 9)
                )
                for r in records
            ]

        assert normalized(wide) == normalized(narrow)

    def test_returns_every_plottable_record_type_and_a_table(
        self, tmp_path: Path
    ) -> None:
        make_example_data(tmp_path, n_folders=1)
        model_dir = tmp_path / "models" / "0"

        records = example_record_generator(model_dir)

        types = {type(r) for r in records}
        assert types == {
            Format2D,
            Trace2D,
            Point2D,
            AxLine,
            HistogramEntry,
            TableEntry,
        }

    def test_table_entries_are_tagged_with_this_runs_directory_name(
        self, tmp_path: Path
    ) -> None:
        make_example_data(tmp_path, n_folders=1)
        model_dir = tmp_path / "models" / "0"

        records = example_record_generator(model_dir)

        table_entries = [r for r in records if isinstance(r, TableEntry)]
        assert table_entries
        assert all(entry.row == "0" for entry in table_entries)
