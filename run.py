#!/usr/bin/env python3
from os.path import join
from sys import argv
from tabulate import tabulate
import pandas as pd
from sklearn.model_selection import train_test_split
from rs_metrics import mapr
from src.models import popular_model, svd_model


class DataSet:
    """
        Class for receiving, processing and splitting a dataset
    """
    def __init__(self, root_dir: str = 'data/raw'):
        self.interactions = pd.read_csv(join(root_dir, 'interactions.csv'))
        self.item_asset = pd.read_csv(join(root_dir, 'item_asset.csv'))
        self.item_price = pd.read_csv(join(root_dir, 'item_price.csv'))
        self.item_subclass = pd.read_csv(join(root_dir, 'item_subclass.csv'))
        self.user_age = pd.read_csv(join(root_dir, 'user_age.csv'))
        self.user_region = pd.read_csv(join(root_dir, 'user_region.csv'))

    def preprocess(self):
        self.interactions.columns = ['user_id', 'item_id', 'action_id']
        self.interactions = self.interactions[['user_id', 'item_id']]

    def get_train_test(self, test_size: float = 0.2):
        return train_test_split(self.interactions, test_size=test_size)


def measure(pred: pd.DataFrame, true: pd.DataFrame, name: str, df: pd.DataFrame = None):
    """
        measure predicted recommendations

        Parameters
        ----------
        pred: pandas dataframe with columns [user_id, item_id, relevance]
            predicted recommendations
        true: pandas dataframe with columns [user_id, item_id]
            true interactions
        name: name of model to put into result table
        df: pandas dataframe of results to concat

        Returns
        -------
        pandas dataframe with measurement results
        """
    if df is None:
        df = pd.DataFrame(columns=['map@10'])
    df.loc[name, 'map@10'] = mapr(true=true, pred=pred, k=10)
    return df


def main(experiment_name):
    dataset = DataSet()
    dataset.preprocess()
    train_data, test_data = dataset.get_train_test()
    test_users = test_data['user_id'].drop_duplicates()

    rec = popular_model.PopularModel()
    rec.fit(train_data)
    rec_pred = rec.predict(log=train_data, users=test_users, k=10, filter_seen=True)
    metrics = measure(rec_pred, test_data, 'popular')

    svd_rec = svd_model.SvdModel(rank=64)
    svd_rec.fit(train_data)
    svd_pred = svd_rec.predict(log=train_data, users=test_users, k=10)
    metrics = measure(svd_pred, test_data, 'svd_rec', metrics)
    metrics.sort_values('map@10', ascending=False, inplace=True)
    experiment_name += '.csv'
    metrics.to_csv(join('reports/experiments', experiment_name), index_label='Model')
    print(tabulate(metrics, headers='keys', tablefmt='fancy_grid'))


if __name__ == '__main__':
    main(argv[1])
