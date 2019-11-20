import csv
import matplotlib.pyplot as plt
import pandas as pd
#from collections import defaultdict
#auto_dict = lambda: defaultdict(auto_dict)


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

    # TODO sort

    return xs, ys


def plot_functions(xs, ys):
    for x, y in zip(xs, ys):
        plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    exp_csv_path = "../experiments/sample.csv"
    df = pd.read_csv(exp_csv_path, sep=";")
    df.join()

    # (tempo) cpu seq vs cpu omp vs gpu (#esempi)
    xs, ys = df_to_ticks(df, independent="dataset_size",
                         dependent=[lambda row: "time" if row["alg_version"] == "seq" else None,
                                    lambda row: "time" if row["alg_version"] == "gpu" else None,
                                    lambda row: "time" if row["alg_version"] == "omp" else None],
                         where=lambda row: row["num_threads"] == 0)

    plot_functions(xs, ys)

    # (speedup) cpu seq vs cpu omp vs gpu (#esempi)
    # TODO
    # xs, ys = df_to_ticks(df, independent="dataset_size",
    #                      dependent=[])


    # omp 1 -> 12
    # TODO

    # nvidia ....TODO