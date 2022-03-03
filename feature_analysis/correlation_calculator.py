import numpy as np
import pandas as pd
from typing import Union


class CorrelationCalculator:
    """
    Class to calculate correlations between features.
    """
    @staticmethod
    def contingency_table(
            feature1: pd.Series,
            feature2: pd.Series,
            normalize: bool = False
    ) -> pd.DataFrame:
        """
        Returns contingency table off passed series.

        :param feature1: Series with data for first feature
        :param feature2: Series with data for second feature
        :param normalize: Flag indication whether contingency table contains counts or distributions
        :return Contingency table as dataframe
        """
        return pd.crosstab(feature1, feature2, normalize=normalize)

    def yules(
            self,
            feature1: pd.Series,
            feature2: pd.Series
    ) -> Union[float, None]:
        """
        Ermittelt Assoziationsmaß nach Yule für zwei dichotome Merkmale. Dies ist ein normiertes Kontingenzmaß.
        Ein A > 0.5 wird als ausgeprägte statistische Kontingenz interpretiert.
        """
        if len(feature1.unique()) != 2 or len(feature2.unique()) != 2:
            print('Please pass two dichotomous features')
            return
        ct = self.contingency_table(
            feature1=feature1,
            feature2=feature2,
            normalize=False
        )
        return (np.sqrt(ct[1][1] * ct[2][2]) - np.sqrt(ct[1][2] * ct[2][1])) / (
                    np.sqrt(ct[1][1] * ct[2][2]) + np.sqrt(ct[1][2] * ct[2][1]))

    @staticmethod
    def n_expected(
            n: int,
            n_j: int,
            n_k: int
    ) -> float:
        """
        Calculates expected count.

        :param n: Total sum of all observations
        :param n_j: Sum of values in row j
        :param n_k: Sum of values in column k
        :return: Expected value count
        """
        return (n_j * n_k) / n

    def cramers_v(
            self,
            feature1: pd.Series,
            feature2: pd.Series
    ) -> float:
        """
        Ermittelt Cramers V als Kontingenmaß zweier nominaler Features. Ist normiert auf 0 <= v <= 1
        """
        ct = self.contingency_table(
            feature1=feature1,
            feature2=feature2,
            normalize=False
        )
        m = np.min([len(feature1.unique()), len(feature2.unique())])
        n = ct.values.sum()

        chi2 = 0
        for j in range(ct.shape[0]):
            for k in range(ct.shape[1]):
                n_jk = ct.iloc[j].iloc[k]
                n_j = ct.iloc[j].sum()
                n_k = ct.iloc[:, k].sum()
                n_jk_e = self.n_expected(
                    n=n,
                    n_j=n_j,
                    n_k=n_k
                )
                chi2 += (np.square(n_jk - n_jk_e) / n_jk_e)
        return np.sqrt(chi2 / (n * (m - 1)))
