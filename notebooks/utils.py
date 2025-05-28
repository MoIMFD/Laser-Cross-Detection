import pickle
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def calculate_distance(p1, p2):
    assert len(p1) == len(p2)
    return np.linalg.norm(np.subtract(p1, p2))


def save_result_dict(result: dict, result_path: Path) -> None:
    if not result_path.parent.exists():
        result_path.parent.mkdir()

    with open(result_path, "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_result_dict(result_path: Path) -> dict:
    if not result_path.exists():
        raise FileExistsError(f"No results stored at {result_path!s}")
    else:
        with open(result_path, "rb") as handle:
            result = pickle.load(handle)
        return result


def save_plot(save_path: Path, fig=None, dpi=600):
    if not fig:
        fig = plt.gcf()
    if not save_path.parent.exists():
        save_path.parent.mkdir()

    fig.savefig(save_path, dpi=dpi)


# create a color map for visualizing the beams
color_list = ["#39ff14", "black", "#c4feb9", "#189b00"]
color_list = ["black", "#00724e", "#00eb58", "#00c7d3", "#00ff70", "#80ffff"]
laser_cmap = mcolors.LinearSegmentedColormap.from_list("", color_list)
