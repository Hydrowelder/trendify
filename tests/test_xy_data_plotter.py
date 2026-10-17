"""Tests for XYDataPlotter's batched drawing of traces and axlines."""

from collections.abc import Iterator
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from trendify.base.pen import Pen
from trendify.formats.format2d import Format2D
from trendify.generator.xy_data_plotter import XYDataPlotter
from trendify.plotting.axline import AxLine, LineOrientation
from trendify.plotting.figure import SingleAxisFigure
from trendify.plotting.trace import Trace2D
from trendify.styling.legend import Legend, LegendLocation
from trendify.styling.marker import Marker

TAG = "tag"


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def _trace(i: int, **pen_kwargs) -> Trace2D:
    return Trace2D(tags=[TAG], x=[0, 1, 2], y=[i, i + 1, i + 2], pen=Pen(**pen_kwargs))


def _plot(traces=(), axlines=()) -> SingleAxisFigure:
    return XYDataPlotter.handle_points_and_traces(
        tag=TAG, points=[], traces=list(traces), axlines=list(axlines), scatters=[]
    )


def _old_plot(traces=(), axlines=()) -> SingleAxisFigure:
    """The behavior before batching: one artist per record."""
    saf = _plot()
    for trace in traces:
        trace.plot_to_ax(saf.ax)
    for axline in axlines:
        axline.plot_to_ax(saf.ax)
    return saf


def _visible_artists(ax) -> list[Artist]:
    return [a for a in ax.get_children() if a.get_visible()]


class TestBatchedTraces:
    def test_many_traces_become_one_collection(self) -> None:
        saf = _plot(_trace(i) for i in range(50))

        collections = [c for c in saf.ax.collections if isinstance(c, LineCollection)]
        assert len(collections) == 1
        assert len(collections[0].get_segments()) == 50

    def test_each_zorder_gets_its_own_collection_with_that_zorder(self) -> None:
        saf = _plot([_trace(0, zorder=1), _trace(1, zorder=5), _trace(2, zorder=1)])

        collections = sorted(
            (c for c in saf.ax.collections if isinstance(c, LineCollection)),
            key=lambda c: c.get_zorder(),
        )
        assert [c.get_zorder() for c in collections] == [1, 5]
        assert [len(c.get_segments()) for c in collections] == [2, 1]

    def test_each_line_keeps_its_own_style(self) -> None:
        saf = _plot(
            [
                _trace(0, color="red", size=3, alpha=0.5, linestyle=":"),
                _trace(1, color="blue", size=1),
            ]
        )

        [collection] = saf.ax.collections
        assert isinstance(collection, LineCollection)
        assert np.asarray(collection.get_linewidth()).tolist() == [3, 1]
        assert np.asarray(collection.get_colors()).tolist() == [
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 1.0],
        ]

    def test_traces_with_markers_are_drawn_individually(self) -> None:
        marked = Trace2D(
            tags=[TAG],
            x=[0, 1],
            y=[0, 1],
            pen=Pen(label="marked"),
            marker=Marker(symbol="o"),
        )
        saf = _plot([_trace(0), marked])

        assert len(saf.ax.collections) == 1
        assert [line.get_label() for line in saf.ax.lines if line.get_visible()] == [
            "marked"
        ]

    def test_trace_without_a_line_is_drawn_individually(self) -> None:
        saf = _plot([_trace(0, linestyle=None, label="dots")])

        assert not saf.ax.collections
        assert [line.get_label() for line in saf.ax.lines] == ["dots"]

    def test_no_traces_adds_nothing(self) -> None:
        saf = _plot()

        assert not saf.ax.collections
        assert not saf.ax.lines


class TestMatchesUnbatchedBehavior:
    TRACES: ClassVar[list[Trace2D]] = [
        _trace(0, label="a", color="red"),
        _trace(5, label="b", color="blue", linestyle="--"),
        _trace(2, label="a", color="green"),
    ]
    AXLINES: ClassVar[list[AxLine]] = [
        AxLine(
            tags=[TAG],
            value=7.0,
            orientation=LineOrientation.HORIZONTAL,
            pen=Pen(label="h"),
        ),
        AxLine(
            tags=[TAG],
            value=-1.0,
            orientation=LineOrientation.VERTICAL,
            pen=Pen(label="v"),
        ),
    ]

    def test_axis_limits(self) -> None:
        new = _plot(self.TRACES, self.AXLINES)
        old = _old_plot(self.TRACES, self.AXLINES)

        assert new.ax.get_xlim() == pytest.approx(old.ax.get_xlim())
        assert new.ax.get_ylim() == pytest.approx(old.ax.get_ylim())

    def test_limits_on_log_axes(self) -> None:
        traces = [Trace2D(tags=[TAG], x=[0, 1, 10, 100], y=[1, 10, 100, 1000])]
        new, old = _plot(traces), _old_plot(traces)
        for saf in (new, old):
            saf.ax.set_xscale("log")
            saf.ax.set_yscale("log")
            saf.ax.autoscale_view()

        assert new.ax.get_xlim() == pytest.approx(old.ax.get_xlim())
        assert new.ax.get_ylim() == pytest.approx(old.ax.get_ylim())

    def test_legend_has_one_entry_per_label(self) -> None:
        new = _plot(self.TRACES, self.AXLINES)
        old = _old_plot(self.TRACES, self.AXLINES)

        def entries(saf: SingleAxisFigure) -> dict[str, Line2D]:
            handles, labels = saf.ax.get_legend_handles_labels()
            return {
                label: handle
                for label, handle in zip(labels, handles, strict=True)
                if isinstance(handle, Line2D)
            }

        assert sorted(entries(new)) == sorted(entries(old)) == ["a", "b", "h", "v"]
        # the last pen with a shared label wins, as before
        assert entries(new)["a"].get_color() == entries(old)["a"].get_color() == "green"

    def test_axlines_span_the_axes_and_autoscale_only_across(self) -> None:
        saf = _plot(axlines=self.AXLINES)

        saf.ax.figure.canvas.draw()
        ymin, ymax = saf.ax.get_ylim()
        xmin, xmax = saf.ax.get_xlim()
        assert ymin <= 7.0 <= ymax
        assert xmin <= -1.0 <= xmax
        # A horizontal line must not push the x limits out to its axes-fraction 0..1 span.
        horizontal_only = _plot(axlines=[self.AXLINES[0]])
        assert (
            horizontal_only.ax.get_xlim()
            == _old_plot(axlines=[self.AXLINES[0]]).ax.get_xlim()
        )

    def test_best_legend_location_still_avoids_the_data(self) -> None:
        # `loc="best"` searches Line2D geometry only. The invisible stand-in line is what lets
        # it still see batched traces, so the legend must land where it did before batching.
        rising = [
            Trace2D(
                tags=[TAG],
                x=list(range(20)),
                y=[v + i for v in range(20)],
                pen=Pen(label="rising"),
            )
            for i in range(10)
        ]
        format2d = Format2D(tags=[TAG], legend=Legend(loc=LegendLocation.BEST))

        extents = []
        for saf in (_plot(rising), _old_plot(rising)):
            saf.apply_format(format2d)
            saf.fig.canvas.draw()
            legend = saf.ax.get_legend()
            assert legend is not None
            extents.append(legend.get_window_extent().extents)

        assert np.allclose(extents[0], extents[1], atol=1.0)

    def test_stand_in_line_is_never_drawn(self) -> None:
        saf = _plot(self.TRACES, self.AXLINES)

        stand_ins = [a for a in saf.ax.lines if a.get_label() == "_nolegend_"]
        assert stand_ins
        assert not any(a.get_visible() for a in stand_ins)
