import pandas as pd
import numpy as np
import os, sys, argparse

def run(file_name):

    df = pd.read_csv(file_name)

    df1 = (df["jorg_train_mean"] <= df["pp_train_mean"])
    df2 = (df["jorg_train_std"] <= df["pp_train_std"])
    df3 = (df["jorg_test_mean"] <= df["pp_test_mean"])
    df4 = (df["jorg_test_std"] <= df["pp_test_std"])

    print("df1: ", df1.value_counts())
    print("df2 : ", df2.value_counts())
    print("df3: ", df3.value_counts())
    print("df4: ", df4.value_counts())

    df1_inds = df.index[df1].tolist()
    df2_inds = df.index[df2].tolist()
    df3_inds = df.index[df3].tolist()
    df4_inds = df.index[df4].tolist()


    print(df1_inds)
    print(df2_inds)
    print(df3_inds)
    print(df4_inds)

    intersection = list(set(df1_inds) & set(df2_inds) & set(df3_inds) & set(df4_inds))
    print(intersection)

    new_l = []

    if len(intersection) == 0:

        intersection = list(set(df1_inds) & set(df3_inds))

    for i in range(len(intersection)):
        n, d, lamb = df.iloc[intersection[i]]["n"], df.iloc[intersection[i]]["d"], df.iloc[intersection[i]]["lamb"]
        l = [int(n), int(d), lamb]
        new_l.append(l)

    print(sorted(new_l))

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", default="plevel_34_43_23_result.csv", type=str)
    args = parser.parse_args()
    
    run(args.file_name)

