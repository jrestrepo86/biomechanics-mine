#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

try:
    from mine.mine import Mine
except ImportError:
    from .mine.mine import Mine


def plot(ax, Rho, teo_mi, mi, label):
    ax.plot(Rho, teo_mi, ".k", label="True")
    ax.plot(Rho, mi, "b", label=label)
    ax.legend(loc="upper center")
    ax.set_title(label)


def testMine():
    # Net
    loss1 = "mine_biased"
    loss2 = "mine"
    loss3 = "remine"
    model_params = {"hidden_dim": 50, "num_hidden_layers": 3, "afn": "elu"}

    N = 5000
    mu = np.array([0, 0])
    Rho = np.linspace(-0.99, 0.99, 21)
    mi_teo = np.zeros(*Rho.shape)
    mi_mine_biased = np.zeros(*mi_teo.shape)
    mi_mine = np.zeros(*mi_teo.shape)
    mi_remine = np.zeros(*mi_teo.shape)

    # Training
    batch_size = "full"
    max_epochs = 100000
    train_params = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "val_size": 0.2,
        "lr": 1e-3,
        "lr_factor": 0.5,
        "lr_patience": 100,
        "stop_patience": 300,
        "stop_min_delta": 0.0,
        "verbose": False,
    }

    for i, rho in enumerate(tqdm(Rho)):
        # Generate data
        cov_matrix = np.array([[1, rho], [rho, 1]])
        joint_samples_train = np.random.multivariate_normal(
            mean=mu, cov=cov_matrix, size=(N, 1)
        )
        X = np.squeeze(joint_samples_train[:, :, 0])
        Y = np.squeeze(joint_samples_train[:, :, 1])

        # Teoric value
        mi_teo[i] = -0.5 * np.log(1 - rho**2)
        # models
        model_biased = Mine(X, Y, loss=loss1, **model_params)
        model_mine = Mine(X, Y, loss=loss2, alpha=0.01, **model_params)
        model_remine = Mine(
            X, Y, loss=loss3, regWeight=0.1, targetVal=0, **model_params
        )

        # Train models
        model_biased.fit(**train_params)
        model_mine.fit(**train_params)
        model_remine.fit(**train_params)

        # Get mi estimation
        mi_mine_biased[i] = model_biased.get_mi()
        mi_mine[i] = model_mine.get_mi()
        mi_remine[i] = model_remine.get_mi()

    # Plot
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    plot(axs[0], Rho, mi_teo, mi_mine_biased, label=loss1)
    plot(axs[1], Rho, mi_teo, mi_mine, label=loss2)
    plot(axs[2], Rho, mi_teo, mi_remine, label=loss3)
    axs[0].set_xlabel("rho")
    axs[0].set_ylabel("mi")
    axs[1].set_xlabel("rho")
    axs[2].set_xlabel("rho")
    fig.suptitle("Mutual information neural estimation", fontsize=13)
    plt.show()


if __name__ == "__main__":
    testMine()
