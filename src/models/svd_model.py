from typing import Optional
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from src.models import parent_model


class SvdModel(parent_model.MyParentModel):
    """
        SVD model class
    """
    user_vectors: np.array
    singular_values: np.array
    item_vectors: np.array

    def __init__(self, rank: int):
        self.rank = rank

    def fit(self,
            log: pd.DataFrame,
            user_features: Optional[pd.DataFrame] = None,
            item_features: Optional[pd.DataFrame] = None) -> 'SVDModel':
        csr_train = csr_matrix((np.ones(log.shape[0]).astype(float),
                                (log['user_id'], log['item_id'])))
        self.user_vectors, self.singular_values, self.item_vectors = svds(A=csr_train, k=self.rank)
        return self


    def _split_pair(self, pred):
        """
        Split column of tuples <item_id, relevance>
        """
        pred['item_id'] = pred['recs'].apply(lambda x: x[0])
        pred['relevance'] = pred['recs'].apply(lambda x: x[1])
        pred.drop(columns=['recs'], inplace=True)

    def _predict(self,
                 log: pd.DataFrame,
                 users: pd.Series,
                 k: int,
                 user_features: Optional[pd.DataFrame] = None,
                 item_features: Optional[pd.DataFrame] = None,
                 filter_seen: bool = True) -> pd.DataFrame:
        def als_pred_by_user(user_id):
            """
            Get top-k recs for user as a list of tuples with <item_id, relevance>
            """
            rel = self.user_vectors[user_id, :] @ np.diag(self.singular_values) @ self.item_vectors
            ids = np.argpartition(rel, -k)[-k:]
            return list(zip(ids, rel[ids]))

        pred = pd.DataFrame(users, columns=['user_id'])
        pred['recs'] = pred['user_id'].apply(als_pred_by_user)
        pred = pred.explode('recs')
        self._split_pair(pred=pred)
        return pred
