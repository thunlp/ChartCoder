import random

# packages
# packages = ['matplotlib==3.8.0', 'pandas==2.1.2', 'textwrap']
packages = ['matplotlib==3.8.0']

plot_styles = [
    'default',
    'classic',
    'Solarize_Light2',
    'dark_background',
    'ggplot',
    'fivethirtyeight',
    'fast',
    'bmh'
]

# tab20 palette
colors = [
    "#F0F8FF",
    "#FAEBD7",
    "#00FFFF",
    "#7FFFD4",
    "#F0FFFF",
    "#F5F5DC",
    "#FFE4C4",
    "#000000",
    "#FFEBCD",
    "#0000FF",
    "#8A2BE2",
    "#A52A2A",
    "#DEB887",
    "#5F9EA0",
    "#7FFF00",
    "#D2691E",
    "#FF7F50",
    "#6495ED",
    "#FFF8DC",
    "#DC143C",
    "#00FFFF",
    "#00008B",
    "#008B8B",
    "#B8860B",
    "#A9A9A9",
    "#006400",
    "#BDB76B",
    "#9932CC",
    "#8B0000",
    "#E9967A",
    "#8FBC8F",
    "#483D8B",
    "#2F4F4F",
    "#00CED1",
    "#9400D3",
    "#FF1493",
    "#00BFFF",
    "#696969",
    "#1E90FF",
    "#B22222",
]

chart_types = ['pie', 'line', 'bar', 'bar_num', "3d", "area", "box", "bubble", 
               "candlestick", "funnel", "heatmap", "multi-axes", "radar", "ring", "rose", 
               "treemap", "violin", "scatter", "quiver","inset", "counter"]
# chart_types = ['bar']

grid_visibility = [True, False]

grid_line_styles = ['-', '--', '-.', ':']

line_styles = ['solid', 'dashed', 'dotted', 'dashdot']

marker_styles = [".", ",", "v", "^", "<", ">", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"]

bar_styles = ['grouped', 'stacked']

bar_arrangement = ['vertical', 'horizontal']

font_types = ['serif', 'sans-serif', 'monospace']

tick_label_styles = ['sci', 'scientific', 'plain']

hatch = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

legend_positions = {
    "upper-right": {
        "loc": 2,  # upper left
        "bbox_to_anchor": "(1.1, 1.1)"
    },
    "lower-right": {
        "loc": 3,  # lower left
        "bbox_to_anchor": "(1.1, -0.1)"
    },
    "upper-left": {
        "loc": 1,  # upper right
        "bbox_to_anchor": "(-0.1, 1.1)"
    },
    "lower-left": {
        "loc": 4,  # lower right
        "bbox_to_anchor": "(-0.1, -0.1)"
    }
}


def legend_position():
    return random.choice(list(legend_positions.keys()))
