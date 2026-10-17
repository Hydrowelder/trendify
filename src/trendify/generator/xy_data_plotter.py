"""
Draws `Point2D`/`Trace2D`/`AxLine` records onto a matplotlib figure. Fed directly by
`RecordStore` query results, so there's no directory-loading state to manage.
"""

from __future__ import annotations

import logging

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.transforms import Transform
from pydantic import BaseModel

from trendify.base.helpers import Tag
from trendify.base.pen import Pen
from trendify.plotting.axline import AxLine, LineOrientation
from trendify.plotting.figure import SingleAxisFigure
from trendify.plotting.point import Point2D
from trendify.plotting.scatter import Scatter2D
from trendify.plotting.trace import Trace2D

__all__ = ["XYDataPlotter"]

logger = logging.getLogger(__name__)


def _nan_separated(segments: list[np.ndarray]) -> np.ndarray:
    """
    Stacks `(N, 2)` vertex arrays into one array with a NaN row after each, so a single
    `Line2D` can stand in for many separate polylines.
    """
    gap = np.full((1, 2), np.nan)
    return np.concatenate([part for segment in segments for part in (segment, gap)])


def _add_batched_lines(
    ax: Axes,
    segments: list[np.ndarray],
    pens: list[Pen],
    transform: Transform | None = None,
) -> None:
    """
    Draws many lines as one `LineCollection` per z-order instead of one `Line2D` each. Every
    artist costs matplotlib transform, clipping and draw work, which dominates when a tag
    holds thousands of lines. Lines keep their given order within a z-order, so overlap is
    unchanged. Each line keeps its own color, width, linestyle and alpha.

    A collection is not a `Line2D`, so two things `Line2D`s used to provide are restored here:
    - **Legend entries**: one empty, labeled `Line2D` per distinct label (the last pen with that
      label wins, as when handles were de-duplicated by label).
    - **Data limits and `loc="best"` legend placement**: both are computed from `Line2D`
      geometry, and matplotlib ignores `LineCollection`s there. One invisible `Line2D` tracing
      every segment carries the geometry, so the axes autoscale and the legend avoids the data
      exactly as before. It is never drawn.

    Args:
        ax (Axes): axes to draw onto
        segments (list[np.ndarray]): one `(N, 2)` array of vertices per line
        pens (list[Pen]): style of each line, parallel to `segments`
        transform (Transform | None): transform for the vertices; `None` means data coordinates

    """
    by_zorder: dict[float, list[int]] = {}
    for i, pen in enumerate(pens):
        by_zorder.setdefault(pen.zorder, []).append(i)
    for zorder, indices in by_zorder.items():
        collection = LineCollection(
            [segments[i] for i in indices],
            colors=[to_rgba(pens[i].color, pens[i].alpha) for i in indices],
            linewidths=[pens[i].size for i in indices],
            linestyles=[pens[i].linestyle for i in indices],
            zorder=zorder,
        )
        if transform is not None:
            collection.set_transform(transform)
        ax.add_collection(collection, autolim=False)

    for pen in {pen.label: pen for pen in pens if pen.label is not None}.values():
        ax.plot([], [], **pen.as_scatter_plot_kwargs())

    vertices = _nan_separated(segments)
    stand_in = Line2D(vertices[:, 0], vertices[:, 1], visible=False, label="_nolegend_")
    if transform is not None:
        stand_in.set_transform(transform)
    ax.add_line(stand_in)


def _plot_traces(ax: Axes, traces: list[Trace2D]) -> None:
    """
    Plots traces, batching the plain ones (no marker, visible line) with `_add_batched_lines`.
    Traces with markers or no line are drawn individually afterward, so at an equal z-order
    they paint over the batched lines.
    """
    plain = [t for t in traces if t.marker is None and t.pen.linestyle is not None]
    if plain:
        _add_batched_lines(
            ax, [np.column_stack([t.x, t.y]) for t in plain], [t.pen for t in plain]
        )
    for trace in traces:
        if trace.marker is not None or trace.pen.linestyle is None:
            trace.plot_to_ax(ax)


def _plot_axlines(ax: Axes, axlines: list[AxLine]) -> None:
    """
    Plots axis lines, batching each orientation with `_add_batched_lines`. Vertices use the
    same blended transform `Axes.axhline`/`axvline` use (axes fraction along the line, data
    value across it), so the lines span the full axes and only the across-axis limit autoscales.
    """
    plain = [a for a in axlines if a.pen.linestyle is not None]
    horizontal = [a for a in plain if a.orientation == LineOrientation.HORIZONTAL]
    vertical = [a for a in plain if a.orientation == LineOrientation.VERTICAL]
    if horizontal:
        _add_batched_lines(
            ax,
            [np.array([[0.0, a.value], [1.0, a.value]]) for a in horizontal],
            [a.pen for a in horizontal],
            ax.get_yaxis_transform(which="grid"),
        )
    if vertical:
        _add_batched_lines(
            ax,
            [np.array([[a.value, 0.0], [a.value, 1.0]]) for a in vertical],
            [a.pen for a in vertical],
            ax.get_xaxis_transform(which="grid"),
        )
    for axline in axlines:
        if axline.pen.linestyle is None:
            axline.plot_to_ax(ax)


class XYDataPlotter(BaseModel):
    """
    Draws `Point2D`/`Scatter2D`/`Trace2D`/`AxLine` records sharing a tag onto a matplotlib
    axes.
    """

    @classmethod
    def handle_points_and_traces(
        cls,
        tag: Tag,
        points: list[Point2D],
        traces: list[Trace2D],
        axlines: list[AxLine],
        scatters: list[Scatter2D],
        saf: SingleAxisFigure | None = None,
    ) -> SingleAxisFigure:
        """
        Plots points (scattered, grouped by marker), scatters, traces, and axlines onto `saf`.

        Args:
            tag (Tag): tag these records belong to (used only if a new figure is created)
            points (list[Point2D]): points to scatter, grouped by `Marker` (each distinct
                marker becomes one `ax.scatter` call/series)
            traces (list[Trace2D]): traces to plot
            axlines (list[AxLine]): axis lines to plot
            scatters (list[Scatter2D]): bulk scatter arrays to plot, each already carrying
                its own single shared `Marker`
            saf (SingleAxisFigure | None): figure to draw onto; a new one is created if `None`

        Returns:
            (SingleAxisFigure): the figure drawn onto

        """
        if saf is None:
            saf = SingleAxisFigure.new(tag=tag)

        logger.debug(
            f"Plotting {len(points)} point(s), {len(traces)} trace(s), {len(axlines)} "
            f"axline(s), {len(scatters)} scatter(s) for {tag = }"
        )
        if points:
            markers = set(p.marker for p in points)
            for marker in markers:
                matching_points = [p for p in points if p.marker == marker]
                x = [p.x for p in matching_points]
                y = [p.y for p in matching_points]
                if x and y:
                    if marker is not None:
                        saf.ax.scatter(x, y, **marker.as_scatter_plot_kwargs())
                    else:
                        saf.ax.scatter(x, y)

        for scatter in scatters:
            scatter.plot_to_ax(saf.ax)

        _plot_traces(saf.ax, traces)
        _plot_axlines(saf.ax, axlines)
        saf.ax.autoscale_view()

        return saf
