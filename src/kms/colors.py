import numpy as np

TABLEAU20 = {
    'blue': np.array([0.122, 0.467, 0.706]),
    'orange': np.array([1.000, 0.498, 0.055]),
    'green': np.array([0.173, 0.627, 0.173]),
    'red': np.array([0.839, 0.153, 0.157]),
    'purple': np.array([0.580, 0.404, 0.741]),
    'brown': np.array([0.549, 0.337, 0.294]),
    'pink': np.array([0.890, 0.467, 0.761]),
    'gray': np.array([0.498, 0.498, 0.498]),
    'yellow': np.array([0.737, 0.741, 0.133]),
    'cyan': np.array([0.090, 0.745, 0.812]),
    'light_blue': np.array([0.267, 0.447, 0.769]),
    'light_orange': np.array([0.992, 0.553, 0.235]),
    'light_green': np.array([0.459, 0.647, 0.125]),
    'light_red': np.array([0.894, 0.102, 0.110]),
    'light_purple': np.array([0.702, 0.486, 0.769]),
    'light_brown': np.array([0.651, 0.337, 0.157]),
    'light_pink': np.array([0.969, 0.588, 0.478]),
    'light_gray': np.array([0.600, 0.600, 0.600]),
    'light_yellow': np.array([0.859, 0.859, 0.553]),
    'light_cyan': np.array([0.267, 0.816, 0.855])
}

TABLEAU20_LIST = [
    TABLEAU20['blue'],
    TABLEAU20['orange'],
    TABLEAU20['green'],
    TABLEAU20['red'],
    TABLEAU20['purple'],
    TABLEAU20['brown'],
    TABLEAU20['pink'],
    TABLEAU20['gray'],
    TABLEAU20['yellow'],
    TABLEAU20['cyan'],
    TABLEAU20['light_blue'],
    TABLEAU20['light_orange'],
    TABLEAU20['light_green'],
    TABLEAU20['light_red'],
    TABLEAU20['light_purple'],
    TABLEAU20['light_brown'],
    TABLEAU20['light_pink'],
    TABLEAU20['light_gray'],
    TABLEAU20['light_yellow'],
    TABLEAU20['light_cyan']
]

RENDER_COLORS = {
    "gray": [0.5, 0.5, 0.5],
    "white": [1, 1, 1, 1],
    "black": [0, 0, 0, 1],
}


def get_color_by_index(idx: int) -> np.ndarray:
    return TABLEAU20_LIST[idx % len(TABLEAU20_LIST)]
