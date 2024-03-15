import numpy as np
import polars as pl
from tqdm import tqdm


class Solver:
    def __init__(self):
        self.reset()

    def reset(self):
        self.df = pl.read_parquet("preproc.parquet")

    def run_full_game(self, answer, progress_bar=False):
        guess_count = 0
        while True:
            guess_count += 1
            guess = self.get_best_guess(progress_bar=progress_bar)
            if guess == answer:
                break
            # TODO: "unvectorize" this for this single case
            outcomes_df = self.get_champion_outcomes(guess)
            filtered_df = outcomes_df.filter(pl.col("name") == answer)
            outcome = filtered_df.to_numpy()[0, 1:].astype(float)
            self.update_df_with_guess(guess, outcome)

        return guess_count

    def get_best_guess(self, progress_bar=True):
        entropies = {}
        for champion in tqdm(self.df["name"], disable=not progress_bar):
            entropies[champion] = self.get_champion_entropy(champion)
        guess = max(entropies, key=entropies.get)
        return guess

    def get_champion_entropy(self, champion):
        outcomes_df = self.get_champion_outcomes(champion)
        outcomes = outcomes_df.to_numpy()[:, 1:].astype(float)

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

        return outcomes_df

    def update_df_with_guess(self, guess, outcome):
        outcome = np.array(outcome).astype(float)

        outcomes_df = self.get_champion_outcomes(guess)
        outcomes_matrix = outcomes_df.to_numpy()[:, 1:].astype(float)
        outcomes_dict = {
            outcomes_df["name"][i]: outcomes_matrix[i]
            for i in range(len(outcomes_matrix))
        }
        filtered_dict = {
            k: v for k, v in outcomes_dict.items() if np.all(v == outcome)
        }
        self.df = self.df.filter(pl.col("name").is_in(filtered_dict.keys()))

    def filter_outcome(self, outcomes_dict, outcome, update_df=True):
        filtered_champs = [k for k, v in outcomes_dict.items() if v == outcome]
        df = self.df.filter(pl.col("name").is_in(filtered_champs))
        if update_df:
            self.df = df
            return filtered_champs
        else:
            return df

    @staticmethod
    def _get_outcome(guess, answer):
        pass

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
