from typing import Optional
import pandas as pd
from src.models import parent_model


class PopularModel(parent_model.MyParentModel):
    """
        Base stupid popular model class
    """
    items_popularity: pd.DataFrame

    def fit(self,
            log: pd.DataFrame,
            user_features: Optional[pd.DataFrame] = None,
            item_features: Optional[pd.DataFrame] = None) -> 'PopularModel':
        self.items_popularity = log.groupby('item_id')['user_id'].count().\
                                    sort_values(ascending=False)\
                                    .rename('popularity') / log['user_id'].nunique()
        return self

    def _predict(self,
                 log: pd.DataFrame,
                 users: pd.Series,
                 k: int,
                 user_features: Optional[pd.DataFrame] = None,
                 item_features: Optional[pd.DataFrame] = None,
                 filter_seen: bool = True) -> pd.DataFrame:
        users_to_join = pd.DataFrame(users)
        users_to_join['key'] = 0
        pred = pd.DataFrame({'item_id': self.items_popularity[:k].index.to_list(),
                             'relevance': self.items_popularity[:k].values})
        pred['key'] = 0
        return users_to_join.merge(pred, on='key', how='outer').drop(columns=['key'])\
            .reset_index(drop=True)
