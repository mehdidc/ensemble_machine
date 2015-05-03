from __future__ import division

import itertools

import numpy as np

from bokeh.plotting import ColumnDataSource, figure, output_file, show
from bokeh.models import HoverTool


def plot_model_embeddings(models, Z, title="untitled", output="out.html"):
    # Create a set of tools to use
    TOOLS="pan,wheel_zoom,box_zoom,reset,hover"
    x = Z[:, 0]
    y = Z[:, 1]
    radii = 1.
    inds = np.arange(1, Z.shape[0] + 1)
    colors = inds
    names = [name for name, model in models]
    # We need to put these data into a ColumnDataSource
    source = ColumnDataSource(
        data=dict(
                x=x,
                y=y,
                radius=radii,
                colors=colors,
                name=names
            )
    )

    output_file(output)

    p = figure(title=title, tools=TOOLS)

    # This is identical to the scatter exercise, but adds the 'source' parameter
    p.circle(x, y, radius=radii, source=source,
                    fill_color=colors, fill_alpha=0.6, line_color=None)

    # EXERCISE (optional) add a `text` renderer to display the index of each circle
    # inside the circle
    p.text(x, y, text=names, alpha=0.5, text_font_size="5pt",
                text_baseline="middle", text_align="center")

    # EXERCISE: try other "marker-like" renderers besides `circle`

    # We want to add some fields for the hover tool to interrogate, but first we
    # have to get ahold of the tool. We can use the 'select' method to do that.
    hover = p.select(dict(type=HoverTool))

    # EXERCISE: add some new tooltip (name, value) pairs. Variables from the
    # data source are available with a "@" prefix, e.g., "@x" will display the
    # x value under the cursor. There are also some special known values that
    # start with "$" symbol:
    #   - $index     index of selected point in the data source
    #   - $x, $y     "data" coordinates under cursor
    #   - $sx, $sy   canvas coordinates under cursor
    #   - $color     color data from data source, syntax: $color[options]:field_name
    # NOTE: tooltips will show up in the order they are in the list
    hover.tooltips = [
        # add to this
        ("(x,y)", "($x, $y)"),
        ("radius", "@radius"),
        ("name", "@name"),
    ]
    show(p)
