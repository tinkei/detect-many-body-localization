import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

dpi = 100
fig_w = 1280
fig_h = 640



def visualize_dataset_rho(dataset, num=5):

    sample_idx = np.random.randint(0, len(dataset), size=num*num)

    fig, axes = plt.subplots(num, num, figsize=(fig_w/dpi,fig_h/dpi*2), dpi=dpi, squeeze=False)

    for i, idx in enumerate(sample_idx):
        image = np.abs(dataset[idx]["image"].squeeze(axis=0))
        axes[i%num,i//num].imshow(image)
        axes[i%num,i//num].annotate('W={:3.1f}'.format(dataset[idx]["W"]), (0.5,0.5), xycoords='axes fraction', ha='center', color='w')

    for axe in axes:
        for ax in axe:
            # ax.legend(loc='best')
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    fig.tight_layout()


def visualize_dataset_H(dataset, num=5):

    sample_idx = np.random.randint(0, len(dataset), size=num*num)

    fig, axes = plt.subplots(num, num, figsize=(fig_w/dpi,fig_h/dpi*2), dpi=dpi, squeeze=False)

    for i, idx in enumerate(sample_idx):
        image = dataset[idx]["image"][0,:,:]
        axes[i%num,i//num].imshow(image, vmin=-10, vmax=10)
        axes[i%num,i//num].annotate('W={:3.1f}'.format(dataset[idx]["W"]), (0.5,0.5), xycoords='axes fraction', ha='center', color='w')

    for axe in axes:
        for ax in axe:
            # ax.legend(loc='best')
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    fig.tight_layout()
