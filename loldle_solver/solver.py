import numpy as np
import polars as pl
from tqdm import tqdm


class Solver:
    def __init__(self):
        self.df = pl.read_parquet("preproc.parquet")
        self.all_releases = sorted(self.df["release"].unique())

    def get_champion_entropy(self, champion):
        outcomes = self.get_champion_outcomes(champion)

        _, counts = np.unique(outcomes, return_counts=True, axis=0)
        probs = counts / counts.sum()

        return calc_entropy(probs)

    def get_champion_outcomes(self, champion):
        champion_row = self.df.filter(pl.col("name") == champion).row(
            0,
            named=True,
        )

        outcomes_df = self.df.with_columns(
            gender=self._vec_get_col_outcome(
                "gender",
                champion_row["gender"],
                is_list_col=False,
            ),
            position=self._vec_get_col_outcome(
                "position",
                champion_row["position"],
                is_list_col=True,
            ),
            species=self._vec_get_col_outcome(
                "species",
                champion_row["species"],
                is_list_col=True,
            ),
            resource=self._vec_get_col_outcome(
                "resource",
                champion_row["resource"],
                is_list_col=False,
            ),
            range=self._vec_get_col_outcome(
                "range",
                champion_row["range"],
                is_list_col=True,
            ),
            region=self._vec_get_col_outcome(
                "region",
                champion_row["region"],
                is_list_col=True,
            ),
            release=self._vec_get_col_outcome(
                "release",
                champion_row["release"],
                is_list_col=False,
            ),
        )

        return outcomes_df.to_numpy()[:, 1:].astype(float)

    def filter_outcome(self, outcomes_dict, outcome, update_df=True):
        filtered_champs = [k for k, v in outcomes_dict.items() if v == outcome]
        df = self.df.filter(pl.col("name").is_in(filtered_champs))
        if update_df:
            self.df = df
            return filtered_champs
        else:
            return df

    @staticmethod
    def _vec_get_col_outcome(col_name, guess_value, is_list_col):
        if col_name == "release":
            return (
                pl.when(pl.col(col_name) > guess_value)
                .then(1)
                .when(pl.col(col_name) < guess_value)
                .then(-1)
                .otherwise(0)
            )
        if is_list_col:
            return (
                pl.when(pl.col(col_name) == guess_value)
                .then(1)
                .when(
                    pl.col(col_name).map_elements(
                        lambda x: any(item in guess_value for item in x)
                    )
                )
                .then(0.5)
                .otherwise(0)
            )
        else:
            return (
                pl.when(pl.col(col_name) == guess_value).then(1).otherwise(0)
            )


def calc_entropy(probs):
    return -np.sum(probs * np.log2(probs))


if __name__ == "__main__":
    solver = Solver()

    entropies = {}
    for champion in tqdm(solver.df["name"]):
        entropies[champion] = solver.get_champion_entropy(champion)
    entropies = {
        k: v
        for k, v in sorted(
            entropies.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    }

    for idx, key in enumerate(entropies):
        if idx == 10:
            break
        print(f"{key}: {entropies[key]:.2f}")


def has_common_elements_with_given(list_from_column, given_list):
    intersection = pl.Series(list_from_column).arr.intersect(
        pl.Series(given_list)
    )
    return len(intersection) > 0
