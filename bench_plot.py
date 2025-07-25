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
sedov_weak, sedov_strong, sedov_flood, \
    maclaurin_weak, maclaurin_strong, maclaurin_flood, \
    crtest_weak, crtest_strong, crtest_flood, \
    sedov_core1, sedov_core_phy, sedov_core_all, \
    maclaurin_core1, maclaurin_core_phy, maclaurin_core_all, \
    crtest_core1, crtest_core_phy, crtest_core_all, \
    _bench_types = list(range(19))
make_prep, make_11, make_1n, make_2n, make_4n, make_8n = list(range(6))

amm = ["avg", "min", "max"]

fig_lab_pos = (0.5, 0.1)  # (0.5, 0.8) for top placement

# Define regex patterns as constants
SEDOV_WEAK_PATTERN = re.compile(r"(.*)sedov, weak")
SEDOV_STRONG_PATTERN = re.compile(r"(.*)sedov, strong")
SEDOV_FLOOD_PATTERN = re.compile(r"(.*)sedov, flood")
MACLAURIN_WEAK_PATTERN = re.compile(r"(.*)maclaurin, weak")
MACLAURIN_STRONG_PATTERN = re.compile(r"(.*)maclaurin, strong")
MACLAURIN_FLOOD_PATTERN = re.compile(r"(.*)maclaurin, flood")
CRTEST_WEAK_PATTERN = re.compile(r"(.*)crtest, weak")
CRTEST_STRONG_PATTERN = re.compile(r"(.*)crtest, strong")
CRTEST_FLOOD_PATTERN = re.compile(r"(.*)crtest, flood")
SEDOV_CORE1_PATTERN = re.compile(r"(.*)Core profiling on sedov, single core")
SEDOV_CORE_PHY_PATTERN = re.compile(r"(.*)Core profiling on sedov, all physical cores")
SEDOV_CORE_ALL_PATTERN = re.compile(r"(.*)Core profiling on sedov, all threads")
MACLAURIN_CORE1_PATTERN = re.compile(r"(.*)Core profiling on maclaurin, single core")
MACLAURIN_CORE_PHY_PATTERN = re.compile(r"(.*)Core profiling on maclaurin, all physical cores")
MACLAURIN_CORE_ALL_PATTERN = re.compile(r"(.*)Core profiling on maclaurin, all threads")
CRTEST_CORE1_PATTERN = re.compile(r"(.*)Core profiling on crtest problem, single core")
CRTEST_CORE_PHY_PATTERN = re.compile(r"(.*)Core profiling on crtest problem, all physical cores")
CRTEST_CORE_ALL_PATTERN = re.compile(r"(.*)Core profiling on crtest problem, all threads")

# Define constants for column indices
MAKE_TIME_INDEX = -4
MAKE_LOAD_INDEX = -1
CRTEST_DATA_COLUMN = 3
SEDOV_DATA_COLUMN = 6
MACLAURIN_DATA_COLUMN = 5
INVALID_COLUMN = -1


# Extract make time and load from columns
def extr_make_t(columns: List[str]) -> Tuple[float, float]:
    """
    Extracts make time and load from the given columns.

    Args:
        columns (List[str]): List of columns from the input file.

    Returns:
        Tuple[float, float]: A tuple containing the make time and load.
    """
    return float(columns[MAKE_TIME_INDEX].replace(',', '.')), float(columns[MAKE_LOAD_INDEX].replace(',', '.').replace('%', ''))


def determine_benchmark_type(line: str) -> int:
    """
    Determines the benchmark type based on the line content.

    Args:
        line (str): Line from the input file.

    Returns:
        int: Benchmark type.
    """
    patterns = {
        SEDOV_WEAK_PATTERN: sedov_weak,
        SEDOV_STRONG_PATTERN: sedov_strong,
        SEDOV_FLOOD_PATTERN: sedov_flood,
        MACLAURIN_WEAK_PATTERN: maclaurin_weak,
        MACLAURIN_STRONG_PATTERN: maclaurin_strong,
        MACLAURIN_FLOOD_PATTERN: maclaurin_flood,
        CRTEST_WEAK_PATTERN: crtest_weak,
        CRTEST_STRONG_PATTERN: crtest_strong,
        CRTEST_FLOOD_PATTERN: crtest_flood,
        SEDOV_CORE1_PATTERN: sedov_core1,
        SEDOV_CORE_PHY_PATTERN: sedov_core_phy,
        SEDOV_CORE_ALL_PATTERN: sedov_core_all,
        MACLAURIN_CORE1_PATTERN: maclaurin_core1,
        MACLAURIN_CORE_PHY_PATTERN: maclaurin_core_phy,
        MACLAURIN_CORE_ALL_PATTERN: maclaurin_core_all,
        CRTEST_CORE1_PATTERN: crtest_core1,
        CRTEST_CORE_PHY_PATTERN: crtest_core_phy,
        CRTEST_CORE_ALL_PATTERN: crtest_core_all
    }
    for pattern, benchmark_type in patterns.items():
        if pattern.match(line):
            return benchmark_type
    return -1


def determine_data_column(benchmark_type: int) -> int:
    """
    Determines the data column based on the benchmark type.

    Args:
        benchmark_type (int): Benchmark type.

    Returns:
        int: Data column index.
    """
    if benchmark_type in (crtest_weak, crtest_strong, crtest_flood, crtest_core1, crtest_core_all, crtest_core_phy):
        return CRTEST_DATA_COLUMN
    elif benchmark_type in (sedov_weak, sedov_strong, sedov_flood, sedov_core1, sedov_core_all, sedov_core_phy):
        return SEDOV_DATA_COLUMN
    elif benchmark_type in (maclaurin_weak, maclaurin_strong, maclaurin_flood, maclaurin_core1, maclaurin_core_all, maclaurin_core_phy):
        return MACLAURIN_DATA_COLUMN
    return INVALID_COLUMN


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
    patterns = {
        "Preparing objects": make_prep,
        "Single-thread make object": make_11,
        "Multi-thread make object": make_1n,
        "Multi-thread make two objects": make_2n,
        "Multi-thread make four objects": make_4n,
        "Multi-thread make eight objects": make_8n,
    }
    for pattern, index in patterns.items():
        if re.match(pattern, line):
            make_real[index], make_load[index] = extr_make_t(columns)
            return True
    return False


MAX_THREADS = 2**20


def process_timing_data(columns: List[str], data: Dict[str, float], timings: Dict[int, List[List[float]]], b_type: int, d_col: int) -> None:
    """
    Processes timing data from the columns and updates the timings dictionary.

    Args:
        columns (List[str]): List of columns from the input file.
        data (Dict[str, float]): Dictionary to store the processed data.
        timings (Dict[int, List[List[float]]]): Dictionary to store timing data.
        b_type (int): Benchmark type.
        d_col (int): Data column index.
    """
    try:
        nthr = int(columns[0])
        if nthr > MAX_THREADS:  # crude protection against eating too much memory due to bad data lines
            logging.warning(f"Ignoring bogus thread number: {columns}")
        elif nthr > 0:
            if nthr not in timings:
                timings[nthr] = [[] for _ in range(_bench_types)]
            timings[nthr][b_type].append(float(columns[d_col]) / (data["big"]**3) if len(columns) >= d_col + 1 else None)
    except ValueError:
        if not re.match("#", columns[0]):
            logging.warning(f"ValueError encountered while processing columns: {columns}")


def set_problem_size_factor(line: str, columns: List[str], data: Dict[str, float]) -> bool:
    """
    Sets the problem size factor based on the line content.

    Args:
        line (str): Line from the input file.
        columns (List[str]): List of columns from the input file.
        data (Dict[str, float]): Dictionary to store the processed data.

    Returns:
        bool: True if the problem size factor was set, False otherwise.
    """
    if re.match("# test domains are scaled by factor of", line):
        data["big"] = float(columns[-1])
        logging.debug(f"Set problem size factor to {data['big']}")
        return True
    return False


def process_line(line: str, columns: List[str], data: Dict[str, float], make_real: List[float], make_load: List[float], timings: Dict[int, List[List[float]]], benchmark_type: int, data_column: int) -> Tuple[int, int]:
    """
    Processes a line from the input file and updates the data structures.

    Args:
        line (str): Line from the input file.
        columns (List[str]): List of columns from the input file.
        data (Dict[str, float]): Dictionary to store the processed data.
        make_real (List[float]): List to store make real times.
        make_load (List[float]): List to store make load values.
        timings (Dict[int, List[List[float]]]): Dictionary to store timing data.
        benchmark_type (int): Benchmark type.
        data_column (int): Data column index.

    Returns:
        Tuple[int, int]: Updated benchmark type and data column index.
    """
    logging.debug(f"Processing line: {line.strip()}")
    if set_problem_size_factor(line, columns, data):
        return benchmark_type, data_column
    elif update_make_times(line, columns, make_real, make_load):
        logging.debug(f"Updated make times for line: {line.strip()}")
    else:
        new_benchmark_type = determine_benchmark_type(line)
        if new_benchmark_type != -1:
            benchmark_type = new_benchmark_type
            data_column = determine_data_column(benchmark_type)
            logging.debug(f"Detected benchmark type: {benchmark_type}, data column: {data_column}")
        if data_column != -1 and len(columns) > 0 and new_benchmark_type == -1:
            process_timing_data(columns, data, timings, benchmark_type, data_column)
    return benchmark_type, data_column


# Read timings from a benchmark file
def initialize_make_times() -> Tuple[List[float], List[float]]:
    """
    Initializes the make times and load lists.

    Returns:
        Tuple[List[float], List[float]]: Initialized make times and load lists.
    """
    make_real: List[float] = [0 for _ in range(make_8n + 1)]
    make_load: List[float] = [0 for _ in range(make_8n + 1)]
    return make_real, make_load


def read_timings(file: str) -> Dict[str, float]:
    """
    Reads timings from a benchmark file and processes the data.

    Args:
        file (str): Path to the benchmark file.

    Returns:
        Dict[str, float]: A dictionary containing the processed data.
    """
    logging.info(f"Reading timings from file: {file}")
    data: Dict[str, float] = {}
    data["filename"] = file
    try:
        with open(file, "r") as f:
            make_real, make_load = initialize_make_times()
            timings: Dict[int, List[List[float]]] = {}
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
def plot_subplot(sub: int, rdata: Dict[str, float], test: int, t_labels: List[str], ld: Dict[str, plt.Line2D], args: argparse.Namespace, fig_lab_pos: Tuple[float, float], exp: float, ntm: int) -> None:
    """
    Plots an individual subplot.

    Args:
        sub (int): Subplot index.
        rdata (Dict[str, float]): Dictionary containing the reduced data.
        test (int): Benchmark test type.
        t_labels (List[str]): List of test labels.
        ld (Dict[str, plt.Line2D]): Dictionary of line objects.
        args (argparse.Namespace): Parsed command-line arguments.
        fig_lab_pos (Tuple[float, float]): Figure label position.
        exp (float): Expansion factor for x-axis limits.
        ntm (int): Maximum number of threads.
    """
    plt.subplot(*PLOT_GRID, sub)
    ym = []
    has_data = False
    for d in rdata:
        n = sorted(rdata[d]["avg"]["timings"].keys())
        y = []
        ymin = []
        ymax = []
        for x in n:
            if test in (crtest_core1, sedov_core1, maclaurin_core1, sedov_core_phy, maclaurin_core_phy, crtest_core_phy, sedov_core_all, maclaurin_core_all, crtest_core_all):
                if rdata[d]["avg"]["timings"][x][test]:
                    y.append(1. / rdata[d]["avg"]["timings"][x][test])
                    if "min" in rdata[d]:
                        ymin.append(1. / rdata[d]["min"]["timings"][x][test])
                        ymax.append(1. / rdata[d]["max"]["timings"][x][test])
                else:
                    y.append(0)
                    if "min" in rdata[d]:
                        ymin.append(0)
                        ymax.append(0)
            else:
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

        if len(y) - y.count(None) > 1:
            plot_with_shading(n, y, ymin if "min" in rdata[d] else None, ymax if "min" in rdata[d] else None,
                              color=ld[d].get_color(), where=ywhere)
            for x in y:
                if x:
                    has_data = True
        else:
            plt.plot(n, y, marker='o')
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

    # Set labels based on test type
    if test in (sedov_flood, maclaurin_flood, crtest_flood):
        xla = "N independent threads"
    elif test in (crtest_core1, sedov_core1, maclaurin_core1, sedov_core_phy, maclaurin_core_phy, crtest_core_phy, sedov_core_all, maclaurin_core_all, crtest_core_all):
        xla = "core number"
    else:
        xla = "N_threads (MPI-1)"

    yla = "steps per second" if test in (crtest_core1, sedov_core1, maclaurin_core1, sedov_core_phy, maclaurin_core_phy, crtest_core_phy, sedov_core_all, maclaurin_core_all, crtest_core_all) else "time [s]"
    if test in (sedov_strong, maclaurin_strong, crtest_strong):
        yla = "time * N_threads [s]"

    add_subplot_labels(t_labels[test], xla, yla)
    setup_axis_ticks(ntm, PLOT_EXPANSION)

    if args.log and has_data:
        plt.yscale("log")
    elif ymax > 0:
        plt.ylim([0., ymax])


# Define test labels as a constant
def get_test_labels(big: float) -> List[str]:
    """
    Generates test labels with appropriate problem sizes.

    Args:
        big (float): Size multiplier for the test domains.

    Returns:
        List[str]: List of formatted test labels.
    """
    base_sizes = {
        'sedov': 64,
        'maclaurin': {'weak': 64, 'strong': 128, 'flood': 64},
        'crtest': 32
    }

    descriptions = {
        'weak': "{}, weak scaling\nN_thr * {} x {} x {}, {} decomposition",
        'strong': "{}, strong scaling\n{} x {} x {}, {} decomposition",
        'flood': "{}, flood scaling, {} x {} x {}",
        'core': "Core performance on {}, {} x {} x {}"
    }

    decomp = {'sedov': 'cartesian', 'maclaurin': 'block', 'crtest': 'noncartesian'}
    labels = []

    # Generate scaling labels
    for test in ['sedov', 'maclaurin', 'crtest']:
        for scale in ['weak', 'strong', 'flood']:
            size = base_sizes[test] if isinstance(base_sizes[test], int) else base_sizes[test][scale]
            size_str = str(int(size * big))
            if scale == 'flood':
                labels.append(descriptions[scale].format(test, size_str, size_str, size_str))
            else:
                labels.append(descriptions[scale].format(test, size_str, size_str, size_str, f"{decomp[test]}"))

    # Generate core performance labels
    for test in ['sedov', 'maclaurin', 'crtest']:
        size = base_sizes[test] if isinstance(base_sizes[test], int) else base_sizes[test]['weak']
        size_str = str(int(size * big))
        labels.extend([descriptions['core'].format(test, size_str, size_str, size_str)] * 3)

    return labels


PLOT_ALPHA = 0.2
PLOT_EXPANSION = 0.25
FIGURE_SIZE = (24, 12)
PLOT_GRID = (4, 3)
SUBPLOT_ADJUSTMENTS = {
    'top': 0.95,
    'left': 0.04,
    'right': 0.99,
    'wspace': 0.15
}


def setup_axis_ticks(ntm: int, exp: float) -> None:
    """Helper function to set up axis ticks consistently"""
    if ntm >= 10:
        x_ticks = list(range(0, ntm, 2 ** (int(m.log2(ntm)) - 2)))
        if ntm not in x_ticks:
            x_ticks.append(ntm)
    else:
        x_ticks = list(range(1, ntm + 1))
    plt.xticks(x_ticks)
    plt.tick_params(axis='y', which='both', right=True)
    plt.xlim(1 - exp, ntm + exp)


def add_subplot_labels(title: str, xlabel: str, ylabel: str) -> None:
    """Helper function to add consistent subplot labels"""
    plt.annotate(title, xy=fig_lab_pos, xycoords="axes fraction", horizontalalignment='center')
    plt.xlabel(xlabel, verticalalignment='center')
    plt.ylabel(ylabel)


def plot_with_shading(x: List[int], y: List[float], ymin: List[float], ymax: List[float],
                      color: str, alpha: float = PLOT_ALPHA, linestyle: str = '-',
                      where: np.ndarray = None) -> None:
    """Helper function to plot a line with optional shading"""
    plt.plot(x, y, color=color, linestyle=linestyle)
    if ymin is not None and ymax is not None:
        linew = 1 if len(x) > 1 else 10
        plt.fill_between(x, ymin, ymax, alpha=alpha, color=color, where=where, linewidth=linew)


def mkrplot(rdata: Dict[str, float], args: argparse.Namespace, output_file: str = None) -> None:
    """
    Plots the benchmark results using matplotlib.

    Args:
        rdata (Dict[str, float]): Dictionary containing the reduced data.
        args (argparse.Namespace): Parsed command-line arguments.
        output_file (str, optional): Path to save the plot. Defaults to None.
    """
    plt.figure(figsize=FIGURE_SIZE)

    # Initialize common parameters
    big = next((d["big"] for d in rdata.values()), -1)
    if any(d["big"] != big for d in rdata.values()):
        print("Mixed benchmark sizes")
        big = 0

    m_labels = ["setup", "serial\nmake", "parallel\nmake", "parallel\nmake 2 obj.", "parallel\nmake 4 obj.", "parallel\nmake 8 obj."]
    t_labels = get_test_labels(big)
    lines = []
    ld = {}

    # Plot compilation time
    plt.subplot(*PLOT_GRID, 1)
    for d in rdata:
        l, = plt.plot(rdata[d]["avg"]["make_real"])
        if "min" in rdata[d]:
            plot_with_shading(list(range(len(rdata[d]["avg"]["make_real"]))),
                              rdata[d]["avg"]["make_real"],
                              rdata[d]["min"]["make_real"],
                              rdata[d]["max"]["make_real"],
                              color=l.get_color())
        lines.append(l)
        ld[d] = l
    add_subplot_labels("compilation time", "", "time [s]")
    plt.xticks(list(range(len(m_labels))), m_labels)
    if args.log and plt.ylim()[0] > 0:
        plt.yscale("log")
    else:
        plt.ylim(ymin=0.)
    plt.xlim(-PLOT_EXPANSION, len(m_labels) - 1 + PLOT_EXPANSION)

    # Plot CPU load
    plt.subplot(*PLOT_GRID, 2)
    for d in rdata:
        plot_with_shading(list(range(len(rdata[d]["avg"]["make_load"]))),
                          rdata[d]["avg"]["make_load"],
                          rdata[d]["min"]["make_load"] if "min" in rdata[d] else None,
                          rdata[d]["max"]["make_load"] if "min" in rdata[d] else None,
                          color=ld[d].get_color())
    add_subplot_labels("compilation CPU usage", "", "CPU load [%]")
    plt.xticks(list(range(len(m_labels))), m_labels)
    plt.ylim(ymin=0.)
    plt.xlim(-PLOT_EXPANSION, len(m_labels) - 1 + PLOT_EXPANSION)

    ntm = 0
    for d in rdata:
        for k in list(rdata[d]["avg"]["timings"].keys()):
            ntm = max(ntm, k)

    def interpolate_missing_points(keys: List[int], values: List[float]) -> List[float]:
        """
        Interpolates missing values (None or 0) with the last valid value from the left.

        Args:
            keys (List[int]): List of x-axis values (core numbers)
            values (List[float]): List of y-axis values

        Returns:
            List[float]: List of interpolated values
        """
        result = []
        last_valid = 0
        for v in values:
            if v:  # if value is valid (not None and not 0)
                last_valid = v
                result.append(v)
            else:
                result.append(last_valid)
        return result

    def plot_core_profile(sorted_keys: List[int], rdata_entry: Dict[str, float], color: str,
                          core_type: int, linestyle: str = '-') -> None:
        """Helper function to plot a single core profile with shading"""
        values = [1. / v if v else 0 for v in [rdata_entry["avg"]["timings"][k][core_type] for k in sorted_keys]]
        if core_type == crtest_core_phy:  # Only interpolate physical cores data
            values = interpolate_missing_points(sorted_keys, values)
        plt.plot(sorted_keys, values, linestyle=linestyle, color=color)

        if "min" in rdata_entry:
            min_vals = [1. / v if v else 0 for v in [rdata_entry["min"]["timings"][k][core_type] for k in sorted_keys]]
            max_vals = [1. / v if v else 0 for v in [rdata_entry["max"]["timings"][k][core_type] for k in sorted_keys]]
            if core_type == crtest_core_phy:
                min_vals = interpolate_missing_points(sorted_keys, min_vals)
                max_vals = interpolate_missing_points(sorted_keys, max_vals)
            plt.fill_between(sorted_keys, min_vals, max_vals, alpha=PLOT_ALPHA, color=color)

    sub = 3
    plt.subplot(4, 3, sub)
    # Create invisible black lines for legend
    legend_lines = []
    legend_lines.append(plt.plot([], [], color='black', label='Single core load')[0])
    legend_lines.append(plt.plot([], [], color='black', linestyle='--', label='Flood all cores')[0])
    legend_lines.append(plt.plot([], [], color='black', linestyle=':', label='Flood all threads')[0])

    # Plot all three data series directly without calling plot_subplot first
    for d in rdata:
        sorted_keys = sorted(rdata[d]["avg"]["timings"].keys())
        color = ld[d].get_color()
        # Plot all three profiles with different line styles
        plot_core_profile(sorted_keys, rdata[d], color, crtest_core1)
        plot_core_profile(sorted_keys, rdata[d], color, crtest_core_phy, linestyle='--')
        plot_core_profile(sorted_keys, rdata[d], color, crtest_core_all, linestyle=':')

    plt.legend(handles=legend_lines)
    plt.xlabel("core number", verticalalignment='center')
    plt.ylabel("steps per second")
    plt.annotate(t_labels[crtest_core1], xy=fig_lab_pos, xycoords="axes fraction", horizontalalignment='center')
    plt.xlim(1 - PLOT_EXPANSION, ntm + PLOT_EXPANSION)
    if args.log and plt.ylim()[0] > 0:
        plt.yscale("log")
    else:
        plt.ylim(ymin=0.)
    if ntm >= 10:
        x_ticks = list(range(0, ntm, 2 ** (int(m.log2(ntm)) - 2)))
        if ntm not in x_ticks:
            x_ticks.append(ntm)
    else:
        x_ticks = list(range(1, ntm + 1))
    plt.xticks(x_ticks)
    plt.tick_params(axis='y', which='both', right=True)

    for test in (sedov_weak, sedov_strong, sedov_flood, maclaurin_weak, maclaurin_strong, maclaurin_flood, crtest_weak, crtest_strong, crtest_flood):
        sub += 1
        plot_subplot(sub, rdata, test, t_labels, ld, args, fig_lab_pos, PLOT_EXPANSION, ntm)

    names = []
    for d in rdata:
        names.append(d)

    plt.subplots_adjust(
        bottom=0.05 + 0.025 * int((len(rdata) - 1) / 2 + 1),
        **SUBPLOT_ADJUSTMENTS
    )
    plt.figlegend((lines), names, loc="lower center", ncol=2, frameon=False)
    plt.annotate("Piernik benchmarks", xy=(0.5, 0.97), xycoords="figure fraction", horizontalalignment='center', size=20)

    if output_file:
        plt.savefig(output_file)
        logging.info(f"Plot saved to {output_file}")
    else:
        plt.show()


# Create a single sample from the data
def initialize_sample(d: Dict[str, float]) -> Dict[str, float]:
    """
    Initializes the sample dictionary for a given data entry.

    Args:
        d (Dict[str, float]): Data entry.

    Returns:
        Dict[str, float]: Initialized sample dictionary.
    """
    sample = {}
    sample["big"] = d["big"]
    sample["weight"] = 1
    for a in amm:
        sample[a] = {}
        sample[a]["timings"] = {}
    for i in ("make_real", "make_load"):
        for a in amm:
            sample[a][i] = deepcopy(d[i])
    for p in d["timings"]:
        for a in amm:
            sample[a]["timings"][p] = []
        for t in d["timings"][p]:
            if len(t) == 0 or None in t:
                for a in amm:
                    sample[a]["timings"][p].append(None)
            else:
                sample["avg"]["timings"][p].append(np.average(t))
                sample["min"]["timings"][p].append(np.min(t))
                sample["max"]["timings"][p].append(np.max(t))
    return sample


def singlesample(data: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Creates a single sample from the given data.

    Args:
        data (List[Dict[str, float]]): List of dictionaries containing the data.

    Returns:
        Dict[str, float]: A dictionary containing the single sample.
    """
    rd: Dict[str, float] = {}
    for d in data:
        d["dname"] = d["filename"]
        rd[d["dname"]] = initialize_sample(d)
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


def update_reduced_values(rd: Dict[str, float], data: Dict[str, float], name: str, d: str) -> None:
    """
    Updates the reduced values for average, min, and max.

    Args:
        rd (Dict[str, float]): Dictionary containing the reduced data.
        data (Dict[str, float]): Dictionary containing the original data.
        name (str): Name of the directory.
        d (str): Data key.
    """
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


# Reduce the data by averaging results from the same directory
def reduce(data: Dict[str, float]) -> Dict[str, float]:
    """
    Reduces the data by averaging results from the same directory.

    Args:
        data (Dict[str, float]): Dictionary containing the data.

    Returns:
        Dict[str, float]: A dictionary containing the reduced data.
    """
    rd: Dict[str, float] = {}
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

        update_reduced_values(rd, data, name, d)
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
    except FileNotFoundError as e:  # message was already printed by validate_files
        exit(1)
    except IOError as e:
        logging.error(f"IO error: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
