# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt


def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    

def cross_validation_visualization_deg(lambds, mse_tr, mse_te, var_tr, var_te, ax, degree):
    """visualization the curves of mse_tr and mse_te."""
    ax.errorbar(lambds, mse_tr, yerr=var_tr, marker=".", color='b', label='train error')
    ax.errorbar(lambds, mse_te, yerr=var_te,  marker=".", color='r', label='test error')
    ax.set_xscale('log')
    ax.set_xlabel("lambda")
    ax.set_ylabel("rmse")
    ax.set_title("cross validation, deg={}".format(degree))
    ax.grid(True)
    ax.legend()
    #plt.savefig("cross_validation")


def bias_variance_decomposition_visualization_box(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    fig, axes = plt.subplots(1,2, figsize=(16,4), sharey=True)
    axes[0].boxplot(rmse_tr)
    axes[0].set_xlabel("degree")
    axes[0].set_ylabel("Train error")
    axes[0].set_title("Bias-Variance Decomposition")
    axes[1].boxplot(rmse_te)
    axes[1].set_xlabel("degree")
    axes[1].set_ylabel("Test error")
    axes[1].set_title("Bias-Variance Decomposition")
    plt.ylim(0.1, 0.7)
    #plt.savefig("bias_variance")

def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    plt.plot(
        degrees,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        label='train',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        label='test',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    plt.ylim(0.1, 0.7)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.title("Bias-Variance Decomposition")
    plt.savefig("bias_variance")
