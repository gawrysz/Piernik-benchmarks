#!/usr/bin/env python

import argparse

sedov_weak, sedov_strong, sedov_flood, maclaurin_weak, maclaurin_strong, maclaurin_flood, crtest_weak, crtest_strong, crtest_flood = range(9)
make_prep, make_11, make_1n, make_2n, make_4n, make_8n = range(6)


def extr_make_t(columns):
    return float(columns[len(columns) - 4]), float(columns[len(columns) - 1].replace('%', ''))


def read_timings(file):

    import re  # overkill, I know

    data = {}
    data["filename"] = file
    with open(file, "r") as f:
        make_real = [0 for x in range(make_8n + 1)]
        make_load = [0 for x in range(make_8n + 1)]
        timings = {}
        data["big"] = 1  # problem size factor

        b_type = -1
        for line in f:
            columns = line.split()
            if re.match("# test domains are scaled by factor of", line):
                data["big"] = int(columns[-1])
            elif re.match("Preparing objects", line):
                make_real[make_prep], make_load[make_prep] = extr_make_t(columns)
            elif re.match("Single-thread make object", line):
                make_real[make_11], make_load[make_11] = extr_make_t(columns)
            elif re.match("Multi-thread make object", line):
                make_real[make_1n], make_load[make_1n] = extr_make_t(columns)
            elif re.match("Multi-thread make two objects", line):
                make_real[make_2n], make_load[make_2n] = extr_make_t(columns)
            elif re.match("Multi-thread make four objects", line):
                make_real[make_4n], make_load[make_4n] = extr_make_t(columns)
            elif re.match("Multi-thread make eight objects", line):
                make_real[make_8n], make_load[make_8n] = extr_make_t(columns)
            else:
                if re.match("(.*)sedov, weak", line):
                    b_type = sedov_weak
                elif re.match("(.*)sedov, strong", line):
                    b_type = sedov_strong
                elif re.match("(.*)sedov, flood", line):
                    b_type = sedov_flood
                elif re.match("(.*)maclaurin, weak", line):
                    b_type = maclaurin_weak
                elif re.match("(.*)maclaurin, strong", line):
                    b_type = maclaurin_strong
                elif re.match("(.*)maclaurin, flood", line):
                    b_type = maclaurin_flood
                elif re.match("(.*)crtest, weak", line):
                    b_type = crtest_weak
                elif re.match("(.*)crtest, strong", line):
                    b_type = crtest_strong
                elif re.match("(.*)crtest, flood", line):
                    b_type = crtest_flood

                if (b_type in (crtest_weak, crtest_strong, crtest_flood)):
                    d_col = 3
                elif (b_type in (sedov_weak, sedov_strong, sedov_flood)):
                    d_col = 6
                elif (b_type in (maclaurin_weak, maclaurin_strong, maclaurin_flood)):
                    d_col = 5
                elif (len(line.strip()) > 1):
                    print "Unknown test: ", line.strip(), b_type
                    exit(1)

            if (len(columns) > 0):
                try:
                    nthr = int(columns[0])
                    if (nthr > 2**20):  # crude protection against eating too much memory due to bad data lines
                        print "Ignoring bogus thread number: ", columns
                    elif (nthr > 0):
                        if (nthr not in timings):
                            timings[nthr] = [[] for x in range(crtest_flood + 1)]
                        timings[nthr][b_type].append(float(columns[d_col]) if (len(columns) >= d_col + 1) else None)
                except ValueError:
                    continue

    data["make_real"] = make_real
    data["make_load"] = make_load
    data["timings"] = timings
    return data


def mkrplot(rdata):
    import matplotlib.pyplot as plt
    import math as m
    import numpy as np

    plt.figure(figsize=(20, 15))

    big = -1
    for d in rdata:
        if (big < 0):
            big = rdata[d]["big"]
        elif (big != rdata[d]["big"]):
            print "Mixed benchmark sizes"
            big = 0

    m_labels = ["setup", "serial\nmake", "parallel\nmake", "parallel\nmake 2 obj.", "parallel\nmake 4 obj.", "parallel\nmake 8 obj."]
    t_labels = [
        "sedov, weak scaling\nN_thr*{} x {} x {}, cartesian decomposition".format(64 * big, 64 * big, 64 * big),
        "sedov, strong scaling\n{} x {} x {}, cartesian decomposition".format(64 * big, 64 * big, 64 * big),
        "sedov, flood scaling\n{} x {} x {}, cartesian decomposition".format(64 * big, 64 * big, 64 * big),
        "maclaurin, weak scaling\nN_thr*{} x {} x {}, block decomposition 32 x 32 x 32".format(64 * big, 64 * big, 64 * big),
        "maclaurin, strong scaling\n{} x {} x {}, block decomposition 32 x 32 x 32".format(128 * big, 128 * big, 128 * big),
        "maclaurin, flood scaling\n{} x {} x {}, block decomposition 32 x 32 x 32".format(64 * big, 64 * big, 64 * big),
        "crtest, weak scaling\nN_thr*{} x {} x {}, noncartesian decomposition".format(32 * big, 32 * big, 32 * big),
        "crtest, strong scaling\n{} x {} x {}, noncartesian decomposition".format(32 * big, 32 * big, 32 * big),
        "crtest, flood scaling\n{} x {} x {}, noncartesian decomposition".format(32 * big, 32 * big, 32 * big)
    ]

    alph = 0.2
    exp = 0.25
    sub = 1
    lines = []
    ld = {}
    plt.subplot(4, 3, sub)
    for d in rdata:
        l, = plt.plot(rdata[d]["avg"]["make_real"])
        if ("min" in rdata[d]):
            plt.fill_between(range(len(rdata[d]["avg"]["make_real"])), rdata[d]["min"]["make_real"], rdata[d]["max"]["make_real"], alpha=alph, color=l.get_color())
        lines.append(l)
        ld[d] = l
    plt.ylabel("time [s]")
    plt.xticks(range(len(rdata[d]["avg"]["make_real"])), m_labels)
    plt.annotate("compilation time", xy=(0.5, 0.1), xycoords="axes fraction", horizontalalignment='center')
    plt.ylim(ymin=0.)
    plt.xlim(-exp, len(m_labels) - 1 + exp)

    sub = 2
    plt.subplot(4, 3, sub)
    for d in rdata:
        plt.plot(rdata[d]["avg"]["make_load"])
        if ("min" in rdata[d]):
            plt.fill_between(range(len(rdata[d]["avg"]["make_load"])), rdata[d]["min"]["make_load"], rdata[d]["max"]["make_load"], alpha=alph, color=ld[d].get_color())
    plt.ylabel("CPU load [%]")
    plt.xticks(range(len(rdata[d]["avg"]["make_load"])), m_labels)
    plt.annotate("compilation CPU usage", xy=(0.5, 0.1), xycoords="axes fraction", horizontalalignment='center')
    plt.ylim(ymin=0.)
    plt.xlim(-exp, len(m_labels) - 1 + exp)

    ntm = 0
    for d in rdata:
        for k in rdata[d]["avg"]["timings"].keys():
            ntm = max(ntm, k)

    sub = 3
    for test in (sedov_weak, sedov_strong, sedov_flood, maclaurin_weak, maclaurin_strong, maclaurin_flood, crtest_weak, crtest_strong, crtest_flood):
        sub += 1
        plt.subplot(4, 3, sub)
        for d in rdata:
            n = rdata[d]["avg"]["timings"].keys()
            y = []
            ymin = []
            ymax = []
            for x in n:
                y.append(rdata[d]["avg"]["timings"][x][test])
                if ("min" in rdata[d]):
                    ymin.append(rdata[d]["min"]["timings"][x][test])
                    ymax.append(rdata[d]["max"]["timings"][x][test])
            if (test in (sedov_strong, maclaurin_strong, crtest_strong)):
                for i in range(len(y)):
                    if (y[i]):
                        y[i] *= n[i]
                        if ("min" in rdata[d]):
                            ymin[i] *= n[i]
                            ymax[i] *= n[i]
            ywhere = np.empty_like(y, dtype=bool)
            if ("min" in rdata[d]):
                for i in range(len(y)):
                    ywhere[i] = ymin[i] and ymax[i]
                    if (not ywhere[i]):
                        ymin[i] = 0.
                        ymax[i] = 0.
            plt.plot(n, y)
            if ("min" in rdata[d]):
                plt.fill_between(n, ymin, ymax, alpha=alph, color=ld[d].get_color(), where=ywhere)
        plt.xlabel("N_threads", verticalalignment='center')
        if (test in (sedov_strong, maclaurin_strong, crtest_strong)):
            plt.ylabel("time * N_threads [s]")
        else:
            plt.ylabel("time [s]")
        plt.annotate(t_labels[test], xy=(0.5, 0.1), xycoords="axes fraction", horizontalalignment='center')
        plt.ylim(ymin=0.)
        plt.xlim(1 - exp, ntm + exp)

        if (ntm >= 10):
            xf, xi = m.modf(m.log10(ntm))
            xf = pow(10, xf)
            if (xf >= 5.):
                xf = 1
                xi += 1
            elif (xf >= 2.):
                xf = 5
            else:
                xf = 2
            xtstep = int(xf * m.pow(10, xi - 1))
            x_ticks = range(0, ntm + xtstep, xtstep)
        else:
            x_ticks = range(1, ntm + 1)
        plt.xticks(x_ticks)

    names = []
    for d in rdata:
        names.append(d)

    plt.subplots_adjust(top=0.95, bottom=0.05 + 0.025 * int((len(rdata) - 1) / 2 + 1), left=0.04, right=0.99, wspace=0.15)
    plt.figlegend((lines), names, loc="lower center", ncol=2, frameon=False)
    plt.annotate("Piernik benchmarks", xy=(0.5, 0.97), xycoords="figure fraction", horizontalalignment='center', size=20)

    plt.show()


def singlesample(data):
    import os.path
    import numpy as np
    amm = ["avg", "min", "max"]
    rd = {}
    for d in data:
        d["dname"] = d["filename"]
        rd[d["dname"]] = {}
        rd[d["dname"]]["big"] = d["big"]
        for a in amm:
            rd[d["dname"]][a] = {}
            rd[d["dname"]][a]["timings"] = {}
        for i in ("make_real", "make_load"):
            for a in amm:
                rd[d["dname"]][a][i] = d[i]
        for p in d["timings"]:
            for a in amm:
                rd[d["dname"]][a]["timings"][p] = []
            for t in d["timings"][p]:
                if (len(t) == 0):
                    for a in amm:
                        rd[d["dname"]][a]["timings"][p].append(None)
                elif (None in t):
                    for a in amm:
                        rd[d["dname"]][a]["timings"][p].append(None)
                else:
                    rd[d["dname"]]["avg"]["timings"][p].append(np.average(t))
                    rd[d["dname"]]["min"]["timings"][p].append(np.min(t))
                    rd[d["dname"]]["max"]["timings"][p].append(np.max(t))
    return rd


def reduce(data):
    import os.path
    import numpy as np
    from copy import deepcopy

    rd = {}
    for d in data:
        d["dname"] = os.path.dirname(d["filename"])
        if (d["dname"] not in rd):
            rd[d["dname"]] = {}
            rd[d["dname"]]["big"] = d["big"]
            rd[d["dname"]]["nt"] = 1
            rd[d["dname"]]["nm"] = 0
            if (np.product(d["make_real"]) * np.product(d["make_load"]) != 0):
                rd[d["dname"]]["nm"] = 1
            rd[d["dname"]]["avg"] = {}
            for i in ("make_real", "make_load", "timings"):
                rd[d["dname"]]["avg"][i] = d[i]
            for i in ("min", "max"):
                rd[d["dname"]][i] = deepcopy(rd[d["dname"]]["avg"])
        else:
            if (rd[d["dname"]]["big"] != d["big"]):
                print "Mixing different problem sizes (" + d["dname"] + ", " + d["filename"] + ")"
                exit(-2)
            if (np.product(d["make_real"]) * np.product(d["make_load"]) != 0):
                rd[d["dname"]]["nm"] += 1
                for i in ("make_real", "make_load"):
                    if (rd[d["dname"]]["nm"] > 1):
                        rd[d["dname"]]["min"][i] = np.minimum(rd[d["dname"]]["min"][i], d[i])
                        rd[d["dname"]]["max"][i] = np.maximum(rd[d["dname"]]["max"][i], d[i])
                        rd[d["dname"]]["avg"][i] = np.add(rd[d["dname"]]["avg"][i], d[i])
                if (rd[d["dname"]]["nm"] == 1):
                    for i in ("make_real", "make_load"):
                        rd[d["dname"]]["avg"][i] = d[i]
                    for i in ("min", "max"):
                        for j in ("make_real", "make_load"):
                            rd[d["dname"]][i][j] = deepcopy(rd[d["dname"]]["avg"][j])
            rd[d["dname"]]["nt"] += 1
            for p in d["timings"]:
                rd[d["dname"]]["min"]["timings"][p] = np.minimum(rd[d["dname"]]["min"]["timings"][p], d["timings"][p])
                rd[d["dname"]]["max"]["timings"][p] = np.maximum(rd[d["dname"]]["max"]["timings"][p], d["timings"][p])
                for i in range(len(rd[d["dname"]]["avg"]["timings"][p])):
                    if (rd[d["dname"]]["avg"]["timings"][p][i] is None or d["timings"][p][i] is None):
                        rd[d["dname"]]["avg"]["timings"][p][i] = None
                    else:
                        rd[d["dname"]]["avg"]["timings"][p][i] = rd[d["dname"]]["avg"]["timings"][p][i] + d["timings"][p][i]

    for d in rd:
        if (rd[d]["nm"] > 1):
            for i in ("make_real", "make_load"):
                rd[d]["avg"][i] /= rd[d]["nm"]
        if (rd[d]["nt"] > 1):
            for p in rd[d]["avg"]["timings"]:
                for i in range(len(rd[d]["avg"]["timings"][p])):
                    if (rd[d]["avg"]["timings"][p][i] is not None):
                        rd[d]["avg"]["timings"][p][i] /= rd[d]["nt"]

    return rd


parser = argparse.ArgumentParser(description='''
Show performance graphs from benchmark files.
By default the data files are grouped by the parent directory.
If there are more files in the same directory, an average is computed and also minimum and maximum values are indicated by shading.
''')
parser.add_argument('file', nargs='+', metavar='benchmark_file', help='file with results obtained by piernik_bench')
parser.add_argument('-s', '--separate', action='store_true', help='do not group the graphs, plot them separately')
args = parser.parse_args()

data = []
for f in args.file:
    data.append(read_timings(f))

rdata = []
if args.separate:
    rdata = singlesample(data)
else:
    rdata = reduce(data)
mkrplot(rdata)
