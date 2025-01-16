#!/usr/bin/env python3

import argparse
from copy import deepcopy
import re  # overkill, I know
import matplotlib.pyplot as plt
import math as m
import numpy as np
import os.path
import logging
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants for benchmark types and make stages
sedov_weak, sedov_strong, sedov_flood, maclaurin_weak, maclaurin_strong, maclaurin_flood, crtest_weak, crtest_strong, crtest_flood = list(range(9))
make_prep, make_11, make_1n, make_2n, make_4n, make_8n = list(range(6))

amm = ["avg", "min", "max"]

fig_lab_pos = (0.5, 0.1)  # (0.5, 0.8) for top placement


# Extract make time and load from columns
def extr_make_t(columns: List[str]) -> Tuple[float, float]:
    """
    Extracts make time and load from the given columns.

    Args:
        columns (List[str]): List of columns from the input file.

    Returns:
        Tuple[float, float]: A tuple containing the make time and load.
    """
    return float(columns[len(columns) - 4].replace(',', '.')), float(columns[len(columns) - 1].replace(',', '.').replace('%', ''))


def determine_benchmark_type(line: str) -> int:
    """
    Determines the benchmark type based on the line content.

    Args:
        line (str): Line from the input file.

    Returns:
        int: Benchmark type.
    """
    if re.match("(.*)sedov, weak", line):
        return sedov_weak
    elif re.match("(.*)sedov, strong", line):
        return sedov_strong
    elif re.match("(.*)sedov, flood", line):
        return sedov_flood
    elif re.match("(.*)maclaurin, weak", line):
        return maclaurin_weak
    elif re.match("(.*)maclaurin, strong", line):
        return maclaurin_strong
    elif re.match("(.*)maclaurin, flood", line):
        return maclaurin_flood
    elif re.match("(.*)crtest, weak", line):
        return crtest_weak
    elif re.match("(.*)crtest, strong", line):
        return crtest_strong
    elif re.match("(.*)crtest, flood", line):
        return crtest_flood
    return -1

def determine_data_column(b_type: int) -> int:
    """
    Determines the data column based on the benchmark type.

    Args:
        b_type (int): Benchmark type.

    Returns:
        int: Data column index.
    """
    if b_type in (crtest_weak, crtest_strong, crtest_flood):
        return 3
    elif b_type in (sedov_weak, sedov_strong, sedov_flood):
        return 6
    elif b_type in (maclaurin_weak, maclaurin_strong, maclaurin_flood):
        return 5
    return -1

def update_make_times(line: str, columns: List[str], make_real: List[float], make_load: List[float]) -> bool:
    """
    Updates make times and load based on the line content.

    Args:
        line (str): Line from the input file.
        columns (List[str]): List of columns from the input file.
        make_real (List[float]): List to store make real times.
        make_load (List[float]): List to store make load values.

    Returns:
        bool: True if make times were updated, False otherwise.
    """
    if re.match("Preparing objects", line):
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
        return False
    return True

def process_timing_data(columns: List[str], data: Dict, timings: Dict, b_type: int, d_col: int) -> None:
    """
    Processes timing data from the columns and updates the timings dictionary.

    Args:
        columns (List[str]): List of columns from the input file.
        data (Dict): Dictionary to store the processed data.
        timings (Dict): Dictionary to store timing data.
        b_type (int): Benchmark type.
        d_col (int): Data column index.
    """
    try:
        nthr = int(columns[0])
        if nthr > 2**20:  # crude protection against eating too much memory due to bad data lines
            logging.warning(f"Ignoring bogus thread number: {columns}")
        elif nthr > 0:
            if nthr not in timings:
                timings[nthr] = [[] for _ in range(crtest_flood + 1)]
            timings[nthr][b_type].append(float(columns[d_col]) / (data["big"]**3) if len(columns) >= d_col + 1 else None)
    except ValueError:
        if not re.match("#", columns[0]):
            logging.warning(f"ValueError encountered while processing columns: {columns}")

def process_line(line: str, columns: List[str], data: Dict, make_real: List[float], make_load: List[float], timings: Dict, b_type: int, d_col: int) -> Tuple[int, int]:
    """
    Processes a line from the input file and updates the data structures.

    Args:
        line (str): Line from the input file.
        columns (List[str]): List of columns from the input file.
        data (Dict): Dictionary to store the processed data.
        make_real (List[float]): List to store make real times.
        make_load (List[float]): List to store make load values.
        timings (Dict): Dictionary to store timing data.
        b_type (int): Benchmark type.
        d_col (int): Data column index.

    Returns:
        Tuple[int, int]: Updated benchmark type and data column index.
    """
    logging.debug(f"Processing line: {line.strip()}")
    if re.match("# test domains are scaled by factor of", line):
        data["big"] = float(columns[-1])
        logging.debug(f"Set problem size factor to {data['big']}")
    elif update_make_times(line, columns, make_real, make_load):
        logging.debug(f"Updated make times for line: {line.strip()}")
    else:
        new_b_type = determine_benchmark_type(line)
        if new_b_type != -1:
            b_type = new_b_type
            d_col = determine_data_column(b_type)
            logging.debug(f"Detected benchmark type: {b_type}, data column: {d_col}")
        if d_col != -1 and len(columns) > 0 and new_b_type == -1:
            process_timing_data(columns, data, timings, b_type, d_col)
    return b_type, d_col

# Read timings from a benchmark file
def read_timings(file: str) -> Dict:
    """
    Reads timings from a benchmark file and processes the data.

    Args:
        file (str): Path to the benchmark file.

    Returns:
        Dict: A dictionary containing the processed data.
    """
    logging.info(f"Reading timings from file: {file}")
    data = {}
    data["filename"] = file
    try:
        with open(file, "r") as f:
            make_real = [0 for x in range(make_8n + 1)]
            make_load = [0 for x in range(make_8n + 1)]
            timings = {}
            data["big"] = 1.  # problem size factor

            b_type = -1
            d_col = -1
            for line in f:
                columns = line.split()
                b_type, d_col = process_line(line, columns, data, make_real, make_load, timings, b_type, d_col)

            data["make_real"] = make_real
            data["make_load"] = make_load
            data["timings"] = timings
    except FileNotFoundError:
        logging.error(f"File not found: {file}")
        exit(1)
    except IOError:
        logging.error(f"Error reading file: {file}")
        exit(1)
    
    return data


# Plot the benchmark results
def mkrplot(rdata: Dict, args: argparse.Namespace, output_file: str = None) -> None:
    """
    Plots the benchmark results using matplotlib.

    Args:
        rdata (Dict): Dictionary containing the reduced data.
        args (argparse.Namespace): Parsed command-line arguments.
        output_file (str, optional): Path to save the plot. Defaults to None.
    """
    plt.figure(figsize=(24, 18))

    big = -1
    for d in rdata:
        if big < 0:
            big = rdata[d]["big"]
        elif big != rdata[d]["big"]:
            print("Mixed benchmark sizes")
            big = 0

    m_labels = ["setup", "serial\nmake", "parallel\nmake", "parallel\nmake 2 obj.", "parallel\nmake 4 obj.", "parallel\nmake 8 obj."]
    t_labels = [
        "sedov, weak scaling\nN_thr * {} x {} x {}, cartesian decomposition".format(int(64 * big), int(64 * big), int(64 * big)),
        "sedov, strong scaling\n{} x {} x {}, cartesian decomposition".format(int(64 * big), int(64 * big), int(64 * big)),
        "sedov, flood scaling, {} x {} x {}".format(int(64 * big), int(64 * big), int(64 * big)),
        "maclaurin, weak scaling\nN_thr * {} x {} x {}, block decomposition 32 x 32 x 32".format(int(64 * big), int(64 * big), int(64 * big)),
        "maclaurin, strong scaling\n{} x {} x {}, block decomposition 32 x 32 x 32".format(int(128 * big), int(128 * big), int(128 * big)),
        "maclaurin, flood scaling\n{} x {} x {}, block decomposition 32 x 32 x 32".format(int(64 * big), int(64 * big), int(64 * big)),
        "crtest, weak scaling\nN_thr * {} x {} x {}, noncartesian decomposition".format(int(32 * big), int(32 * big), int(32 * big)),
        "crtest, strong scaling\n{} x {} x {}, noncartesian decomposition".format(int(32 * big), int(32 * big), int(32 * big)),
        "crtest, flood scaling, {} x {} x {}".format(int(32 * big), int(32 * big), int(32 * big))
    ]

    alph = 0.2
    exp = 0.25
    sub = 1
    lines = []
    ld = {}
    plt.subplot(4, 3, sub)
    for d in rdata:
        l, = plt.plot(rdata[d]["avg"]["make_real"])
        if "min" in rdata[d]:
            plt.fill_between(list(range(len(rdata[d]["avg"]["make_real"]))), rdata[d]["min"]["make_real"], rdata[d]["max"]["make_real"], alpha=alph, color=l.get_color())
        lines.append(l)
        ld[d] = l
    plt.ylabel("time [s]")
    plt.xticks(list(range(len(rdata[d]["avg"]["make_real"]))), m_labels)
    plt.annotate("compilation time", xy=fig_lab_pos, xycoords="axes fraction", horizontalalignment='center')
    if args.log and plt.ylim()[0] > 0:
        plt.yscale("log")
    else:
        plt.ylim(ymin=0.)
    plt.xlim(-exp, len(m_labels) - 1 + exp)

    sub = 2
    plt.subplot(4, 3, sub)
    for d in rdata:
        plt.plot(rdata[d]["avg"]["make_load"])
        if "min" in rdata[d]:
            plt.fill_between(list(range(len(rdata[d]["avg"]["make_load"]))), rdata[d]["min"]["make_load"], rdata[d]["max"]["make_load"], alpha=alph, color=ld[d].get_color())
    plt.ylabel("CPU load [%]")
    plt.xticks(list(range(len(rdata[d]["avg"]["make_load"]))), m_labels)
    plt.annotate("compilation CPU usage", xy=fig_lab_pos, xycoords="axes fraction", horizontalalignment='center')
    plt.ylim(ymin=0.)
    plt.xlim(-exp, len(m_labels) - 1 + exp)

    ntm = 0
    for d in rdata:
        for k in list(rdata[d]["avg"]["timings"].keys()):
            ntm = max(ntm, k)

    sub = 3
    for test in (sedov_weak, sedov_strong, sedov_flood, maclaurin_weak, maclaurin_strong, maclaurin_flood, crtest_weak, crtest_strong, crtest_flood):
        sub += 1
        plt.subplot(4, 3, sub)
        ym = []
        has_data = False
        for d in rdata:
            n = sorted(rdata[d]["avg"]["timings"].keys())
            y = []
            ymin = []
            ymax = []
            for x in n:
                y.append(rdata[d]["avg"]["timings"][x][test])
                if "min" in rdata[d]:
                    ymin.append(rdata[d]["min"]["timings"][x][test])
                    ymax.append(rdata[d]["max"]["timings"][x][test])
            if test in (sedov_strong, maclaurin_strong, crtest_strong):
                for i in range(len(y)):
                    if y[i]:
                        y[i] *= n[i]
                        if "min" in rdata[d]:
                            ymin[i] *= n[i]
                            ymax[i] *= n[i]
            ywhere = np.empty_like(y, dtype=bool)
            if "min" in rdata[d]:
                for i in range(len(y)):
                    ywhere[i] = ymin[i] and ymax[i]
                    if not ywhere[i]:
                        ymin[i] = 0.
                        ymax[i] = 0.
            if len(n) > 1:
                plt.plot(n, y)
                for x in y:
                    if x is not None:
                        has_data = True
            else:
                plt.plot(n, y, marker='o')
            if "min" in rdata[d]:
                linew = 1 if len(n) > 1 else 10
                plt.fill_between(n, ymin, ymax, alpha=alph, color=ld[d].get_color(), where=ywhere, linewidth=linew)
            try:
                ym.append(max(filter(lambda v: v is not None, y)))
            except ValueError:
                pass
        ymax = plt.ylim()[1]
        try:
            if ymax > 1.5 * max(ym):
                ymax = 1.2 * max(ym)
        except ValueError:
            pass

        xla = "N independent threads" if test in (sedov_flood, maclaurin_flood, crtest_flood) else "N_threads (MPI-1)"
        plt.xlabel(xla, verticalalignment='center')
        if test in (sedov_strong, maclaurin_strong, crtest_strong):
            plt.ylabel("time * N_threads [s]")
        else:
            plt.ylabel("time [s]")
        plt.annotate(t_labels[test], xy=fig_lab_pos, xycoords="axes fraction", horizontalalignment='center')
        if args.log and has_data:  # don't crash on empty plots
            plt.yscale("log")
        else:
            plt.ylim([0., ymax])
        plt.xlim(1 - exp, ntm + exp)

        if ntm >= 10:
            xf, xi = m.modf(m.log10(ntm))
            xf = pow(10, xf)
            if xf >= 5.:
                xf = 1
                xi += 1
            elif xf >= 2.:
                xf = 5
            else:
                xf = 2
            xtstep = int(xf * m.pow(10, xi - 1))
            x_ticks = list(range(0, ntm + xtstep, xtstep))
        else:
            x_ticks = list(range(1, ntm + 1))
        plt.xticks(x_ticks)

    names = []
    for d in rdata:
        names.append(d)

    plt.subplots_adjust(top=0.95, bottom=0.05 + 0.025 * int((len(rdata) - 1) / 2 + 1), left=0.04, right=0.99, wspace=0.15)
    plt.figlegend((lines), names, loc="lower center", ncol=2, frameon=False)
    plt.annotate("Piernik benchmarks", xy=(0.5, 0.97), xycoords="figure fraction", horizontalalignment='center', size=20)

    if output_file:
        plt.savefig(output_file)
        logging.info(f"Plot saved to {output_file}")
    else:
        plt.show()


# Create a single sample from the data
def singlesample(data: List[Dict]) -> Dict:
    """
    Creates a single sample from the given data.

    Args:
        data (List[Dict]): List of dictionaries containing the data.

    Returns:
        Dict: A dictionary containing the single sample.
    """
    rd = {}
    for d in data:
        d["dname"] = d["filename"]
        rd[d["dname"]] = {}
        rd[d["dname"]]["big"] = d["big"]
        rd[d["dname"]]["weight"] = 1
        for a in amm:
            rd[d["dname"]][a] = {}
            rd[d["dname"]][a]["timings"] = {}
        for i in ("make_real", "make_load"):
            for a in amm:
                rd[d["dname"]][a][i] = deepcopy(d[i])
        for p in d["timings"]:
            for a in amm:
                rd[d["dname"]][a]["timings"][p] = []
            for t in d["timings"][p]:
                if len(t) == 0 or None in t:
                    for a in amm:
                        rd[d["dname"]][a]["timings"][p].append(None)
                else:
                    rd[d["dname"]]["avg"]["timings"][p].append(np.average(t))
                    rd[d["dname"]]["min"]["timings"][p].append(np.min(t))
                    rd[d["dname"]]["max"]["timings"][p].append(np.max(t))
    return rd


def average_values(weight1: float, value1: float, weight2: float, value2: float) -> float:
    """
    Averages two values with their respective weights.

    Args:
        weight1 (float): Weight of the first value.
        value1 (float): The first value.
        weight2 (float): Weight of the second value.
        value2 (float): The second value.

    Returns:
        float: The weighted average of the two values.
    """
    if value1 * value2 == 0.:
        return 0.
    return (weight1 * value1 + weight2 * value2) / (weight1 + weight2)

# Reduce the data by averaging results from the same directory
def reduce(data: Dict) -> Dict:
    """
    Reduces the data by averaging results from the same directory.

    Args:
        data (Dict): Dictionary containing the data.

    Returns:
        Dict: A dictionary containing the reduced data.
    """
    rd = {}
    for d in data:
        name = os.path.dirname(d)
        if len(name) < 1:
            name = d
        if name not in rd:
            rd[name] = deepcopy(data[d])
            continue

        if data[d]["big"] != rd[name]["big"]:
            print("Mixing different problem sizes (" + d + ", " + name + ")")
            exit(-2)

        for i in ("make_real", "make_load"):
            for v in range(len(rd[name]["avg"][i])):
                rd[name]["avg"][i][v] = average_values(rd[name]["weight"], rd[name]["avg"][i][v], data[d]["weight"], data[d]["avg"][i][v])
                if rd[name]["min"][i][v] == 0:
                    rd[name]["min"][i][v] = data[d]["min"][i][v]
                elif data[d]["min"][i][v] != 0:
                    rd[name]["min"][i][v] = min(rd[name]["min"][i][v], data[d]["min"][i][v])
            rd[name]["max"][i] = np.maximum(rd[name]["max"][i], data[d]["max"][i])

        i = "timings"
        for p in list(rd[name]["avg"][i].keys()):
            if p not in data[d]["avg"][i]:
                for a in amm:
                    del rd[name][a][i][p]
                continue

            for v in range(len(rd[name]["avg"][i][p])):
                if rd[name]["avg"][i][p][v] is None or data[d]["avg"][i][p][v] is None:
                    for a in amm:
                        rd[name][a][i][p][v] = None
                else:
                    rd[name]["avg"][i][p][v] = average_values(rd[name]["weight"], rd[name]["avg"][i][p][v], data[d]["weight"], data[d]["avg"][i][p][v])
                    rd[name]["min"][i][p][v] = min(rd[name]["min"][i][p][v], data[d]["min"][i][p][v])
                    rd[name]["max"][i][p][v] = max(rd[name]["max"][i][p][v], data[d]["max"][i][p][v])

        rd[name]["weight"] += 1

    return rd


def validate_files(files: List[str]) -> None:
    """
    Validates the input files to ensure they exist and are readable.

    Args:
        files (List[str]): List of file paths to validate.

    Raises:
        FileNotFoundError: If any of the files do not exist.
        IOError: If any of the files are not readable.
    """
    for file in files:
        if not os.path.isfile(file):
            logging.error(f"File not found: {file}")
            raise FileNotFoundError(f"File not found: {file}")
        if not os.access(file, os.R_OK):
            logging.error(f"File not readable: {file}")
            raise IOError(f"File not readable: {file}")

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='''
    Show performance graphs from benchmark files.
    By default the data files are grouped by the parent directory.
    If there are more files in the same directory, an average is computed and also minimum and maximum values are indicated by shading.
    ''')
    parser.add_argument('file', nargs='+', metavar='benchmark_file', help='file with results obtained by piernik_bench')
    parser.add_argument('-s', '--separate', action='store_true', help='do not group the graphs according to their directories, plot them separately')
    parser.add_argument('-l', '--log', action='store_true', help='use logarithmic scale for the measured execution time')
    parser.add_argument('-o', '--output', metavar='output_file', help='file to save the plot')
    return parser.parse_args()

def main() -> None:
    """
    Main function to execute the script.
    """
    try:
        # Argument parser setup
        args = parse_arguments()

        # Validate input files
        validate_files(args.file)

        data = []
        for f in args.file:
            data.append(read_timings(f))

        rdata = singlesample(data)
        if not args.separate:
            rdata = reduce(rdata)

        mkrplot(rdata, args, args.output)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        exit(1)

if __name__ == "__main__":
    main()
