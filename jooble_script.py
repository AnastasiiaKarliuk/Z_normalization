# -*- coding: utf-8 -*-
"""Jooble_script

Use case:
	$ python3 jooble_script.py -train data/train.tsv -test data/test.tsv 
"""

import pandas as pd
import numpy as np
import argparse


def transform_solumns(df: pd.DataFrame) -> pd.DataFrame:
    n_feature = df.iloc[0].features.split(',')[0]
    columns = [f'feature_{n_feature}'] + [f'feature_{n_feature}_{i}' for i, _ in enumerate(df.features.iloc[0].split(',')[1:])]
    df[columns] = df.features.str.split(",",expand=True,)
    df = df.drop(columns=['features', f'feature_{n_feature}'], axis=1)
    return df


class Scaling:
    def train_transform(self, data_train: pd.DataFrame) -> pd.DataFrame:
        arr = np.array(data_train, dtype=float)
        self.means, self.stds = arr.mean(axis=0), np.std(arr, axis=0)
        return (arr - self.means) / self.stds

    def test_transform(self, test_data: pd.DataFrame) -> pd.DataFrame:
        arr = np.array(test_data, dtype=float)
        return (arr - self.means) / self.stds


def abs_mean_diff(row: pd.Series, test: pd.DataFrame) -> float:
    curr = np.array(test[test.index==row.name], dtype=float)[0][row['max_feature_2_index']]
    mean =  np.array(test, dtype='float').mean(axis=0)[row['max_feature_2_index']]
    return np.abs(mean-float(curr))


def z_normalization(path_train: str, path_test: str, path_out: str):
    # reading files
    train = pd.read_csv(path_train, sep='\t', index_col=0)
    test = pd.read_csv(path_test, sep='\t', index_col=0)
    
    # transforming columns
    test = transform_solumns(test)
    train = transform_solumns(train)

    # Z-normalization
    scaler = Scaling()
    train_scaled = scaler.train_transform(train)
    test_scaled = scaler.train_transform(test)

    # making final dataFrame
    df_proc = pd.DataFrame({'feature_2_stand':[','.join((test_scaled[i]).astype(str)) for i in range(test.shape[0])], 
                            'max_feature_2_index':np.argmax(test_scaled, axis=1)}, index=test.index)
    df_proc['max_feature_2_abs_mean_diff'] = df_proc.apply(lambda x: abs_mean_diff(x, test), axis=1)
    df_proc.to_csv(path_out, sep='\t')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', type=str)
    parser.add_argument('-test', type=str)
    parser.add_argument('-out', type=str, default='test_proc.tsv')
    args = parser.parse_args()
    
    z_normalization(args.train, args.test, args.out)


if __name__ == '__main__':
    main()
