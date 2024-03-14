import numpy as np
import polars as pl
from tqdm import tqdm


class Solver:
    def __init__(self, compare_release=True):
        self.compare_release = compare_release
        self.df = pl.read_parquet("preproc.parquet")
        self.all_releases = sorted(self.df["release"].unique())

        self.n_variables = (
            len(self.df.columns) - 1
            if compare_release
            else len(self.df.columns)
        )

    def get_champion_entropy(self, champion):
        outcomes, _ = self.get_champion_outcomes(champion)

        _, counts = np.unique(outcomes, return_counts=True, axis=0)
        probs = counts / counts.sum()

        return calc_entropy(probs)

    def get_champion_outcomes(self, champion):
        outcomes = []
        outcomes_dict = {}

        for answer in self.df["name"]:
            outcome = self.get_outcome(champion, answer)
            outcomes.append(outcome)
            outcomes_dict[answer] = outcome
        return np.array(outcomes), outcomes_dict

    def get_outcome(self, guess, answer):
        guess_row = self.df.filter(pl.col("name") == guess).row(0)
        answer_row = self.df.filter(pl.col("name") == answer).row(0)
        col_outcomes = []

        for i in range(1, self.n_variables):
            guess_value = guess_row[i]
            answer_value = answer_row[i]
            col_outcome = self._get_column_outcome(guess_value, answer_value)
            col_outcomes.append(col_outcome)
        if self.compare_release:
            col_outcome = self._get_release_direction(
                guess_value,
                answer_value,
            )
            col_outcomes.append(col_outcome)

        return col_outcomes

    def filter_outcome(self, outcomes_dict, outcome, update_df=True):
        filtered_champs = [k for k, v in outcomes_dict.items() if v == outcome]
        df = self.df.filter(pl.col("name").is_in(filtered_champs))
        if update_df:
            self.df = df
            return filtered_champs
        else:
            return df

    @staticmethod
    def _get_column_outcome(guess_value, answer_value):
        if guess_value == answer_value:
            return 1
        if isinstance(guess_value, list) and (
            set(guess_value) & set(answer_value)
        ):
            return 0.5
        return 0

    @staticmethod
    def _get_release_direction(guess_value, answer_value):
        if guess_value < answer_value:
            return 1
        if guess_value > answer_value:
            return -1
        return 0


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
