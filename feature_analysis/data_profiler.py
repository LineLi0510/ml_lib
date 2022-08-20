import pandas as pd
from typing import List


class DataProfiler:
    def profile_dataframe(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Profiles dataframe.

        :param df: Dataframe to analyse
        :return: Dataframe with profiling information
        """
        df_profiled = pd.DataFrame()
        df_profiled['feature'] = list(df.columns)
        df_profiled['dtype'] = self.dtype(df=df)
        df_profiled['value_count'] = len(df)
        df_profiled['na_count'] = self.na_count(df=df)
        df_profiled['mean'] = self.mean(df=df)
        df_profiled['std'] = self.std(df=df)
        df_profiled['mode'] = self.mode(df=df)
        df_profiled['nunique'] = self.nunique(df=df)
        df_profiled['distinctnes'] = df_profiled['nunique'] / df_profiled['value_count']
        return df_profiled

    @staticmethod
    def na_count(
            df: pd.DataFrame
    ) -> List[int]:
        """
        Checks nan count for every column.

        :param df: Dataframe to analyse.
        :return: Dict with nan count
        """
        return [df[col].isna().sum() for col in df]

    @staticmethod
    def dtype(
            df: pd.DataFrame
    ) -> List[str]:
        """
        Checks nan count for every column.

        :param df: Dataframe to analyse.
        :return: Dict with nan count
        """
        return list(df.dtypes)

    @staticmethod
    def nunique(
            df: pd.DataFrame
    ) -> List[int]:
        """
        Checks nan count for every column.

        :param df: Dataframe to analyse.
        :return: Dict with nan count
        """
        return [df[col].nunique() for col in df]

    @staticmethod
    def mean(
            df: pd.DataFrame
    ) -> List[int]:
        """
        Calculates mean for numerical columns.

        :param df: Dataframe to analyse.
        :return: List with mean for each column
        """
        mean_list = []
        for col in df:
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_list.append(df[col].mean())
            else:
                mean_list.append('-')
        return mean_list

    @staticmethod
    def std(
            df: pd.DataFrame
    ) -> List[int]:
        """
        Calculates mean for numerical columns.

        :param df: Dataframe to analyse.
        :return: List with mean for each column
        """
        mean_list = []
        for col in df:
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_list.append(df[col].std())
            else:
                mean_list.append('-')
        return mean_list

    @staticmethod
    def mode(
            df: pd.DataFrame
    ) -> List[int]:
        """
        Calculates mode for each non float column.

        :param df: Dataframe to analyse.
        :return: Dict with nan count
        """
        mode_list = []
        for col in df:
            if pd.api.types.is_float_dtype(df[col]):
                mode_list.append('-')
            else:
                mode_list.append(df[col].mode()[0])
        return mode_list
