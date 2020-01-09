import argparse
import matplotlib.pyplot as plt
import pandas as pd


def add_speedups(data_frame):
    sequential_time = {}
    for index, row in data_frame.iterrows():
        if row["hw"] == "cpu" and row["num_threads"] == 1:
            sequential_time[row["dataset_size"]] = row["eval_time"]
    speedup_column = []
    for index, row in data_frame.iterrows():
        speedup_column.append(sequential_time[row["dataset_size"]] / row["eval_time"])
    data_frame["speedup"] = speedup_column
    return data_frame


def df_to_ticks(data_frame, independent, dependent, where=lambda x:True, sort=True):

    xs = []
    ys = []
    for _ in dependent:
        xs.append([])
        ys.append([])
    for index, row in data_frame.iterrows():
        if not where(row):
            continue
        for d in range(len(dependent)):
            if not dependent[d](row) is None:
                xs[d].append(row[independent])
                ys[d].append(row[dependent[d](row)])

    for d in range(len(dependent)):
        xs[d], ys[d] = zip(*sorted(zip(xs[d], ys[d]), key=lambda pair: pair[0]))

    return xs, ys


def plot_functions(xs, ys, labels, colors, xlabel, ylabel, mode="plot"):
    for x, y, label, color in zip(xs, ys, labels, colors):
        if mode == "plot":
            plt.plot(x, y, label=label, color=color)
        elif mode == "bar":
            plt.bar(range(len(y)), y, label=label, color=color)
        else:
            print("Unsupported mode")
            return
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if mode == "bar":
        plt.xticks(range(len(ys[0])), xs[0])

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plots from csv.')
    parser.add_argument("exp_csv_path", help="csv to load", default=None)
    parser.add_argument("-s", "--sep", help="value separator in csv", default=";")
    args = parser.parse_args()

    if args.exp_csv_path is None:
        print("Error: no csv file specified")
        exit(1)

    df = pd.read_csv(args.exp_csv_path, sep=args.sep)
    df = add_speedups(df)

    max_dataset_size = df['dataset_size'].max()
    best_gpu_time = df.where(df['dataset_size'] == max_dataset_size).where(df["hw"] == "gpu")["eval_time"].min()
    best_gpu_block = df.where(df['dataset_size'] == max_dataset_size).where(df["hw"] == "gpu").where(df["eval_time"] == best_gpu_time)["num_threads"].max()

    ''' Comparison: CPU_sequential vs CPU_OMP vs GPU '''
    # dataset_size -> time
    xs, ys = df_to_ticks(df, independent="dataset_size",
                         dependent=[lambda row: "eval_time" if row["hw"] == "cpu" and row["num_threads"] == 1 else None,
                                    lambda row: "eval_time" if row["hw"] == "cpu" and row["num_threads"] == 6 else None,
                                    lambda row: "eval_time" if row["hw"] == "gpu" else None],
                         where=lambda row: row["num_threads"] == 1 or row["num_threads"] == 6 or row["num_threads"] == best_gpu_block)
    plot_functions(xs, ys, labels=["SEQ", "OMP", "GPU"], colors=["blue", "orange", "green"], xlabel="dataset size", ylabel="time")
    # dataset_size -> speedup
    xs, ys = df_to_ticks(df, independent="dataset_size",
                         dependent=[lambda row: "speedup" if row["hw"] == "cpu" and row["num_threads"] == 6 else None,
                                    lambda row: "speedup" if row["hw"] == "gpu" else None],
                         where=lambda row: row["num_threads"] == 6 or row["num_threads"] == best_gpu_block)
    plot_functions(xs, ys, labels=["OMP", "GPU"], colors=["orange", "green"], xlabel="dataset size", ylabel="speedup")

    ''' OMP '''
    # num_threads -> time
    xs, ys = df_to_ticks(df, independent="num_threads",
                         dependent=[lambda row: "eval_time"],
                         where=lambda row: row["hw"] == "cpu" and row["dataset_size"] == max_dataset_size)
    plot_functions(xs, ys, labels=["OMP"], colors=["orange"], xlabel="num threads", ylabel="time")
    # num_threads -> speedup
    xs, ys = df_to_ticks(df, independent="num_threads",
                         dependent=[lambda row: "speedup"],
                         where=lambda row: row["hw"] == "cpu" and row["dataset_size"] == max_dataset_size)
    plot_functions(xs, ys, labels=["OMP"], colors=["orange"], xlabel="num threads", ylabel="speedup")

    ''' GPU '''
    # num_threads -> time
    xs, ys = df_to_ticks(df, independent="num_threads",
                         dependent=[lambda row: "eval_time"],
                         where=lambda row: row["hw"] == "gpu" and row["dataset_size"] == max_dataset_size)
    plot_functions(xs, ys, labels=["GPU"], colors=["green"], xlabel="threads per block", ylabel="time", mode="bar")
    # num_threads -> speedup
    xs, ys = df_to_ticks(df, independent="num_threads",
                         dependent=[lambda row: "speedup"],
                         where=lambda row: row["hw"] == "gpu" and row["dataset_size"] == max_dataset_size)
    plot_functions(xs, ys, labels=["GPU"], colors=["green"], xlabel="threads per block", ylabel="speedup", mode="bar")
