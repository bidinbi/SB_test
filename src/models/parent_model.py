import abc
from typing import Optional
from copy import deepcopy
import pandas as pd


def leave_top_k(pred: pd.DataFrame,
                k: int,
                group_by_col: str = 'user_id',
                order_by_col: str = 'relevance') -> pd.DataFrame:
    """
    crop predictions to leave top-k recommendations for each user
    """
    if pred.groupby(group_by_col)[group_by_col].count().max() <= k:
        return pred
    cropped_pred = deepcopy(pred)
    cropped_pred['rank'] = (cropped_pred
                            .groupby(group_by_col)[[order_by_col]]
                            .rank(method="first", ascending=False))
    cropped_pred = cropped_pred[cropped_pred['rank'] <= k].drop(columns=['rank'])
    return cropped_pred


def filter_seen_items(log: pd.DataFrame,
                      pred: pd.DataFrame) -> pd.DataFrame:
    """
    filter pairs `user-item` present in log out of pred
    """
    log_filtered = log[log['user_id'].isin(pred['user_id'].unique())]
    pred = pred.merge(log_filtered[['user_id', 'item_id']].drop_duplicates(),
                      on=['user_id', 'item_id'],
                      how='outer',
                      indicator=True)
    return pred[pred['_merge'] == 'left_only'].drop(columns=['_merge'])


class MyParentModel:
    """
        Base class for create other model
    """
    @abc.abstractmethod
    def fit(self,
            log: pd.DataFrame,
            user_features: Optional[pd.DataFrame] = None,
            item_features: Optional[pd.DataFrame] = None):
        """
        fit recommender

        Parameters
        ----------
        log : pandas dataframe with columns [user_id, item_id, relevance]
        user_features : pandas dataframe with column `user_id` and features columns
        item_features : pandas dataframe with column `item_id` and features columns

        Returns
        -------

        """

    def predict(self,
                log: pd.DataFrame,
                users: pd.Series,
                k: int,
                user_features: Optional[pd.DataFrame] = None,
                item_features: Optional[pd.DataFrame] = None,
                filter_seen: bool = True) -> pd.DataFrame:
        """
        predict with fitted model, filter seen and crop to top-k for each user

        Parameters
        ----------
        log: pandas dataframe with columns [user_id, item_id, relevance]
            used to filter seen and to make predictions by some models
        users: user ids to recommend for
        k: number of recommendations for each user
        user_features: pandas dataframe with column `user_id` and features columns
        item_features: pandas dataframe with column `item_id` and features columns
        filter_seen: if True, items present in user history are filtered from predictions

        Returns
        -------
        pandas dataframe with columns [user_id, item_id, relevance]
            top-k recommended items for each user from `users`.
        """

        # overhead
        max_items_in_train = log.groupby('user_id')[['item_id']].count().max()[0] \
            if filter_seen else 0

        pred = self._predict(
            log=log,
            users=users,
            k=max_items_in_train + k,
            user_features=user_features,
            item_features=item_features,
            filter_seen=filter_seen)
        if filter_seen:
            pred = filter_seen_items(log, pred)

        return leave_top_k(pred, k)

    @abc.abstractmethod
    def _predict(self,
                 log: pd.DataFrame,
                 users: pd.Series,
                 k: int,
                 user_features: Optional[pd.DataFrame] = None,
                 item_features: Optional[pd.DataFrame] = None,
                 filter_seen: bool = True) -> pd.DataFrame:
        """
        predict with fitted model
        """
