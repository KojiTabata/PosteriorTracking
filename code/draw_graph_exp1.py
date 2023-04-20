import re
from itertools import combinations
import datetime
import os.path
import time
import pickle
import random
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
import sys
from multiprocessing import Pool
from policy import get_w_new
from util import get_d, BernoulliArm


PNAMES = {
    "PTracking": "P-Tracking",
    "CTracking": "C-Tracking",
    "DTracking": "D-Tracking",
    "Thompson sampling": "Thompson sampling-CB",
    "UGapE": "UGapE",
    "LUCB": "LUCB",
    "APT": "APT",
    "UCB": "HDoC",
}

PNAMES1 = {
    "PTracking": "P-Tracking",
    "CTracking": "C-Tracking",
    "DTracking": "D-Tracking",
    "Thompson sampling": "Thompson sampling-CB",
}

PNAMES2 = {
    "PTracking": "P-Tracking",
    "UGapE": "UGapE",
    "LUCB": "LUCB",
    "APT": "APT",
    "UCB": "HDoC",
}


def pname_mod(pname):
    return PNAMES.get(pname, pname)


def draw_graph(savedir, fp_pickle, pname_dic, suffix):

    with open(fp_pickle, "rb") as f:
        d = pickle.load(f)

    fn = os.path.basename(fp_pickle)
    basename, ext = os.path.splitext(fn)

    porder = {}
    for i, pname in enumerate(PNAMES):
        porder[pname] = i

    for k1, d1 in d.items():
        delta, theta, K = k1

        print(delta, theta, K)

        fig_dir = f"./{savedir}/fig/{basename}"
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        pnames = []
        for L, d2 in d1.items():
            bars = {}
            ms = set()
            muss = {}
            for m, d3 in d2.items():
                ms.add(m)
                for pname in sorted([pname for pname in d3 if pname in pname_dic], key=lambda pname: porder[pname]):
                    d4 = d3[pname]
                    if pname not in pnames:
                        pnames.append(pname)
                    for sname, d5 in d4.items():
                        if (pname, sname) not in bars:
                            bars[pname, sname] = {}
                        if m not in bars[pname, sname]:
                            bars[pname, sname][m] = {"T": [], "ctime": []}
                        for (mus, i), (T, res, selected, ctime, extra) in d5.items():
                            muss[m] = mus
                            bars[pname, sname][m]["T"].append(T)
                            bars[pname, sname][m]["ctime"].append(ctime)
            ms = list(sorted(ms))
            if len(ms) > 1:
                w = (ms[1] - ms[0]) / (len(bars) + 1)
            else:
                w = 1.0 / (len(bars) + 1)

            cm = plt.get_cmap("tab10")
            xss, yss, css, errs = [], [], [], []
            dict_plot = {}
            for i, (pname, sname) in enumerate(bars):
                xs, ys, cs, err = [], [], [], []
                for m in bars[pname, sname]:
                    xs.append(m + i*w)
                    Ts = bars[pname, sname][m]["T"]
                    ys.append(np.mean(Ts))
                    err.append(np.std(Ts))
                    cs.append(cm(i/10))
                dict_plot[pname] = (xs, ys, cs, err)
                yss += ys
                xss += xs
                css += cs
                errs += err

            lucbs = []
            for y, e in zip(yss, errs):
                lucbs += [y - e, y + e]
            sorted_y = sorted(lucbs, reverse=True)
            max_ = max(lucbs)
            breaks = None
            for y0, y1 in zip(sorted_y[:-1], sorted_y[1:]):
                if y0 < max_ / 2:
                    break
                if 2 * y1 < y0:
                    breaks = (y1 * 1.05, y0 * 0.95)
                    break

            if False and breaks:
                import matplotlib.gridspec as gridspec
                r1, r2 = max_*1.05 - breaks[1], breaks[0]
                r1, r2 = r1 / (r1 + r2), r2 / (r1 + r2)
                gs = gridspec.GridSpec(2, 1, height_ratios=[r1, r2], hspace=0.05)
                ax1 = plt.subplot(gs[0])
                ax2 = plt.subplot(gs[1])

                for pname, (xs, ys, cs, err) in dict_plot.items():
                    ax1.bar(xs, ys, width=w, yerr=err, label=f"{pname_mod(pname)}")
                    ax2.bar(xs, ys, width=w, yerr=err, label=f"{pname_mod(pname)}")

                ax1.set_ylim(breaks[1], max_*1.05)
                ax2.set_ylim(0.0, breaks[0])

                ax1.spines['bottom'].set_visible(False)
                ax1.tick_params('x', length=0, which='major')
                ax2.spines['top'].set_visible(False)

                d = 0.03
                d1, d2 = d * r2, d * r1
                kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
                ax1.plot((0 - d, 0 + d), (0 - d1, 0 + d1), **kwargs)  # top-left diagonal
                ax1.plot((1 - d, 1 + d), (0 - d1, 0 + d1), **kwargs)  # top-right diagonal
                kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
                ax2.plot((0 - d, 0 + d), (1 - d2, 1 + d2), **kwargs)  # bottom-left diagonal
                ax2.plot((1 - d, 1 + d), (1 - d2, 1 + d2), **kwargs)  # bottom-right diagonal

                digits = -int(np.log10(delta))
                ax2.set_xticks(ms)
                ax1.set_xticks(ms)
                ax1.grid()
                ax2.grid()
                ax1.set_xticklabels(["" for m in ms])
            else:
                for pname, (xs, ys, cs, err) in dict_plot.items():
                    plt.bar(xs, ys, width=w, yerr=err, label=f"{pname_mod(pname)}")
                digits = int(np.log10(delta))
                plt.xticks(ms)
                plt.grid()


            plt.xlabel("$M$")
            plt.savefig(f"{fig_dir}/stopping_time_{L}_{suffix}.png", dpi=200)
            plt.close()

        fig, ax = plt.subplots()
        for i, pname in enumerate(sorted([pname for pname in pnames if pname in pname_dic], key=lambda pname: porder[pname])):
            ax.bar([i], [1], label=PNAMES[pname])
        legend = ax.legend(frameon=False, handletextpad=1.0, ncol=1, columnspacing=2.0, bbox_to_anchor=(1.05, 1), loc="upper left")
        legend_fig = legend.figure
        legend_fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted())
        fig.savefig(f"{fig_dir}/legend_{suffix}.png", bbox_inches=bbox, dpi=100)
        plt.close()

        fig, ax = plt.subplots(constrained_layout=False)
        pnames_sub = [pname for pname in pnames if pname in pname_dic]
        for i, pname in enumerate(sorted(pnames_sub, key=lambda pname: porder[pname])):
            ax.bar([i], [1], label=PNAMES[pname])
        legend = ax.legend(frameon=False, handletextpad=1.0, ncol=len(pnames_sub), columnspacing=2.0, bbox_to_anchor=(0.0, -0.05), loc="upper left")
        legend_fig = legend.figure
        legend_fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted())
        fig.savefig(f"{fig_dir}/legend_h_{suffix}.png", bbox_inches=bbox, dpi=100)

        plt.close()


def how_many_times_draw_each_arm(savedir, fp_pickle):
    with open(fp_pickle, "rb") as f:
        d = pickle.load(f)

    fn = os.path.basename(fp_pickle)
    basename, ext = os.path.splitext(fn)

    for k1, d1 in d.items():
        delta, theta, K = k1

        if K < 10:
            continue

        fig_dir = f"./{savedir}/fig/selections_{basename}"
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        for L, d2 in d1.items():
            bars = {}
            ms = set()
            for m, d3 in d2.items():
                ms.add(m)
                for pname, d4 in d3.items():
                    for sname, d5 in d4.items():
                        if (pname, sname) not in bars:
                            bars[pname, sname] = {}
                        if m not in bars[pname, sname]:
                            bars[pname, sname][m] = {"T": [], "ctime": []}

                        max_T = 0
                        has_extra = False
                        for (mus, i), (T, res, selected, ctime, extra) in d5.items():
                            bars[pname, sname][m]["T"].append(T)
                            bars[pname, sname][m]["ctime"].append(ctime)
                            if max_T < T:
                                max_T = T
                            if extra:
                                has_extra = True

                        cm = plt.get_cmap("seismic")
                        im = np.zeros((len(d5), max_T)) + 0.5
                        for (mus, i), (T, res, selected, ctime, extra) in d5.items():
                            im[i, :len(selected)] = [mus[j] for j, x in selected]
                        plt.imshow(im, vmin=0, vmax=1, cmap=cm, aspect=max_T/len(d5), interpolation="none")

                        plt.ylabel("run")
                        plt.xlabel("time round")
                        plt.colorbar()
                        plt.title(f"$K={K},L={L},M={m}$")
                        plt.savefig(f"{fig_dir}/which_arm_{K:03}_{L:02}_{m:02}_{pname_mod(pname)}.png", dpi=300)
                        plt.savefig(f"{fig_dir}/which_arm_{K:03}_{L:02}_{m:02}_{pname_mod(pname)}.pdf")
                        plt.close()

                        if has_extra:
                            print(max_T)
                            im = np.zeros((len(d5), max_T + 1, 3)) + 1.0
                            for (mus, i), (T, res, selected, ctime, extra) in d5.items():
                                for j in range(len(extra)):
                                    if extra[j] == "P":
                                        im[i, j, :] = (1, 0, 0)
                                    else:
                                        im[i, j, :] = (0, 0, 1)
                            plt.title(f"{pname_mod(pname)}")
                            plt.ylabel("run")
                            plt.xlabel("time round")
                            plt.imshow(im, aspect=int(max_T/len(d5)), interpolation="none")
                            plt.savefig(f"{fig_dir}/Posterior_Greedy_{K:03}_{L:02}_{m:02}_{pname_mod(pname)}.png", dpi=300)
                            plt.savefig(f"{fig_dir}/Posterior_Greedy_{K:03}_{L:02}_{m:02}_{pname_mod(pname)}.pdf")
                            plt.close()


if __name__ == "__main__":
    from experiment1 import SAVEDIR

    plt.rcParams['font.family'] = "Times New Roman"
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['xtick.major.width'] = 1.0  # x軸主目盛り線の線幅
    plt.rcParams['ytick.major.width'] = 1.0  # y軸主目盛り線の線幅
    plt.rcParams['font.size'] = 22  # フォントの大きさ
    plt.rcParams['axes.linewidth'] = 1.0  # 軸の線幅edge linewidth。囲みの太さ

    plt.rcParams['figure.constrained_layout.use'] = True

    for pname_dic, suffix in [(PNAMES1, "tracking"), (PNAMES2, "others")]:
        print(suffix)
        for pkl in os.listdir(SAVEDIR):
            print(pkl)
            re_pkl = re.compile(r"\.pkl$")
            mo = re_pkl.search(pkl)
            if mo:
                draw_graph(SAVEDIR, f"{SAVEDIR}/{pkl}", pname_dic, suffix)
                how_many_times_draw_each_arm(SAVEDIR, f"{SAVEDIR}/{pkl}")
