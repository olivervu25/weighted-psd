import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

WPSD_LEVELS = [0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
WPSD_NAMES = [f'wPSD L={L}' for L in WPSD_LEVELS]

PSD_L_LEVELS = [1, 2, 4, 8, 16, 32]
PSD_L_NAMES = [f'PSD L={L}' for L in PSD_L_LEVELS]

BASE_TEST_NAMES = ['RFSD', 'RFSD(alpha)',
                   'IMQ KSD', 'Gauss KSD', 'Gauss FSSD-opt',
                   'PSD r1', 'PSD r2', 'PSD r3', 'PSD r4']

TEST_NAMES = BASE_TEST_NAMES + WPSD_NAMES + PSD_L_NAMES

ORDERED_TEST_NAMES = ['RFSD',
                      'RFSD(alpha)', 'RFSD(RBM)',
                      'Gauss KSD', 'IMQ KSD', 'Gauss FSSD-opt',
                      'PSD r1', 'PSD r2', 'PSD r3', 'PSD r4'
                      ] + WPSD_NAMES + PSD_L_NAMES


def test_name_colors_dict():
    base_colors = sns.color_palette(n_colors=len(BASE_TEST_NAMES))
    wpsd_colors = [tuple(c) for c in
                   plt.cm.viridis(np.linspace(0.15, 0.85, len(WPSD_NAMES)))]
    psd_colors = [tuple(c) for c in
                  plt.cm.plasma(np.linspace(0.15, 0.85, len(PSD_L_NAMES)))]
    d = {}
    d.update(zip(BASE_TEST_NAMES, base_colors))
    d.update(zip(WPSD_NAMES, wpsd_colors))
    d.update(zip(PSD_L_NAMES, psd_colors))
    return d
