import numpy as np
import pandas as pd
from typing import Union


class CorrelationCalculator:
    """
    Sammlung von Methoden zur Ermittlung verschiedener Korrelationsmaße für nominale, ordinale oder kardinale
    Merkmale.

    zwei dichotome Merkmale: yule
    zwei nominale Merkmale: cramers_v
    zwei orinale Merkmale: rang_corr (Rangkorrelation nach Spearman oder Kendall)
    zwei kardinale Merkmale: pearson_corr
    """
    @staticmethod
    def contingency_table(
            feature1: pd.Series,
            feature2: pd.Series,
            normalize: bool = False
    ) -> pd.DataFrame:
        """
        Returns contingency table off passed series.

        :param feature1: Series with data for first nominal or ordinal feature
        :param feature2: Series with data for second nominal or ordinal feature
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

        :param feature1: Series with first dichotome feature
        :param feature2: Series with second dichotome feature
        :return yules correlation coefficient
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

    def cramers_v(
            self,
            feature1: pd.Series,
            feature2: pd.Series
    ) -> float:
        """
        Ermittelt Cramers V als Kontingenmaß zweier nominaler Features. Ist normiert auf 0 <= v <= 1

        :param feature1: Series with data for first feature
        :param feature2: Series with data for second feature
        :return cramers v correlation coefficient
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

    def cramers_v_matrix(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        df_corr = pd.DataFrame(index=df.columns, columns=df.columns)
        for col1 in df:
            for col2 in df:
                cramers_v = self.cramers_v(df[col1], df[col2])
                df_corr[col1].loc[col2] = cramers_v
        return df_corr

    @staticmethod
    def pearson_corr(
            feature1: pd.Series,
            feature2: pd.Series
    ) -> float:
        """
        Korrelationskoeffizient nach Pearson. Misst den linearen Zusammenhang zweier intervallskalierter
        Merkmale. Nicht lineare Zusammenhänge zwischen Merkmalen können nicht ermittelt werden.

        :param feature1: Series with data for first feature
        :param feature2: Series with data for second feature
        :return pearson correlation coefficient
        """
        return feature1.corr(feature2, method='pearson')

    @staticmethod
    def pearson_corr_matrix(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Matrix with Korrelationskoeffizienten nach Pearson. Misst den linearen Zusammenhang zweier intervallskalierter
        Merkmale. Nicht lineare Zusammenhänge zwischen Merkmalen können nicht ermittelt werden.

        :param df: Dataframe to calculate correlation for
        :return matrix with pearson correlation coefficients
        """
        return df.corr(method='pearson')

    @staticmethod
    def rank_corr(
            feature1: pd.Series,
            feature2: pd.Series,
            method: str
    ) -> float:
        """
        Ermittelt Rangkorrelation nach Kendall oder Sprearman für ordinale Merkmale oder kardinaler Merkmale,
        die keinen linearen Zsuammenhang aufweisen
        Kendall tau für Merkmale mit vielen Bindungen, Spearman, wenn wenige Bindungen vorliegen.

        :param feature1: Series with data for first feature
        :param feature2: Series with data for second feature
        :param method: 'kendal' or 'spearman'
        :return pearson correlation coefficient
        """
        return feature1.corr(feature2, method=method)

    @staticmethod
    def rank_corr_matrix(
            df: pd.DataFrame,
            method: str
    ) -> pd.DataFrame:
        """
        Ermittelt Matrix der Rangkorrelationen nach Kendall oder Sprearman für ordinale Merkmale oder
        kardinaler Merkmale, die keinen linearen Zusammenhang aufweisen
        Kendall tau für Merkmale mit vielen Bindungen, Spearman, wenn wenige Bindungen vorliegen.

        :param df: Dataframe to calculate correlation for
        :param method: 'kendal' or 'spearman'
        :return pearson correlation coefficient
        """
        return df.corr(method=method)

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
