import os
import sys

import numpy as np
import polars as pl

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

from loldle_solver.solver import Solver

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
GREEN = (0, 255, 0)
YELLOW = (255, 204, 0)
RED = (255, 0, 0)


class Simulator:
    RESULT_MAP = {
        "name": {None: WHITE},
        "gender": {None: WHITE, 0: RED, 1: GREEN},
        "position": {None: WHITE, 0: RED, 0.5: YELLOW, 1: GREEN},
        "species": {None: WHITE, 0: RED, 0.5: YELLOW, 1: GREEN},
        "resource": {None: WHITE, 0: RED, 1: GREEN},
        "range": {None: WHITE, 0: RED, 0.5: YELLOW, 1: GREEN},
        "region": {None: WHITE, 0: RED, 0.5: YELLOW, 1: GREEN},
        "release": {None: WHITE, -1: RED, 1: YELLOW, 0: GREEN},
    }

    def __init__(self):
        pygame.init()
        self.font = pygame.font.Font(None, 80)
        self.medium_font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 20)
        self.clock = pygame.time.Clock()
        self.width = 1920
        self.height = 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("LoLdle Solver")
        self.table_area = pygame.Rect(
            0.78 * self.width,
            0.125 * self.height,
            0.2 * self.width,
            0.95 * self.height,
        )

        self.color_inactive = WHITE
        self.color_active = pygame.Color("dodgerblue2")
        self.color = self.color_inactive

        self.input_box = pygame.Rect(
            0.33 * self.width,
            0.05 * self.height,
            0.5 * self.width,
            0.1 * self.height,
        )
        self.active = False
        self.text = ""

        self.solver = Solver()
        self.displayed_entries = []
        self.n_possib = len(self.solver.df)
        self.n_possib_entries = []
        self.guesses = self.solver.get_best_guesses()

    def draw_left_header(self):
        text_surf = self.medium_font.render(
            f"Número de possibilidades: {self.n_possib}",
            True,
            WHITE,
        )
        self.screen.blit(
            text_surf,
            (0.05 * self.height, 0.05 * self.height),
        )

    def draw_right_header(self):
        text_surf = self.medium_font.render(
            "Melhores chutes:",
            True,
            WHITE,
        )
        self.screen.blit(
            text_surf,
            (0.78 * self.width, 0.05 * self.height),
        )

    def draw_table(self):
        header_font = self.medium_font
        entry_font = pygame.font.Font(None, 44)

        headers = ["Campeão", "Info. média"]
        header_height = header_font.get_linesize()
        entry_height = entry_font.get_linesize()
        x_pos = self.table_area.x
        y_pos = self.table_area.y

        for header in headers:
            text_surface = header_font.render(header, True, WHITE)
            self.screen.blit(text_surface, (x_pos, y_pos))
            x_pos += self.table_area.width / 2

        y_pos += header_height
        for entry in list(self.guesses.items())[:20]:
            x_pos = self.table_area.x
            for column in entry:
                if isinstance(column, float):
                    column = f"{column:.2f}"
                text_surface = entry_font.render(str(column), True, WHITE)
                self.screen.blit(text_surface, (x_pos, y_pos))
                x_pos += self.table_area.width / 2
            y_pos += entry_height

    def draw_text_input_box(self):
        pygame.draw.rect(self.screen, self.color, self.input_box, 10)
        text_surf = self.font.render(self.text, True, WHITE)
        self.screen.blit(
            text_surf,
            (self.input_box.x + 15, self.input_box.y + 25),
        )
        self.input_box.w = max(
            0.33 * self.width,
            text_surf.get_width() + 30,
        )

    def draw_button(self, text, position, size=(200, 40)):
        button_rect = pygame.Rect(position, size)
        pygame.draw.rect(self.screen, WHITE, button_rect)
        text_surf = self.font.render(text, True, BLACK)
        text_rect = text_surf.get_rect(center=button_rect.center)
        self.screen.blit(text_surf, text_rect)
        return button_rect

    def draw_entries(self):
        y_offset = 0.2 * self.height
        square_size = (125, 125)
        gap = 10

        for idx, entry in enumerate(self.displayed_entries):
            x_offset = 0.33 * self.width - (1.75 * square_size[0])
            for key, value_dict in entry.items():
                square_rect = pygame.Rect(x_offset, y_offset, *square_size)
                if key != "name":
                    self.displayed_entries[idx][key]["rect"] = square_rect
                square = pygame.draw.rect(
                    self.screen,
                    self.RESULT_MAP[key][value_dict["result"]],
                    square_rect,
                    width=5,
                )
                self.draw_multi_line_text(
                    str(value_dict["value"]),
                    square.center,
                    square.width - 20,
                    square.height - 20,
                    line_spacing=10,
                )

                x_offset += square_size[0] + gap

            text_surf = self.medium_font.render(
                f"Possib.: {self.n_possib_entries[idx]}",
                True,
                WHITE,
            )
            self.screen.blit(
                text_surf,
                (0.15 * self.height, y_offset + square.height // 2 - 15),
            )
            y_offset += square_size[1] + gap

    def draw_multi_line_text(
        self,
        text,
        center,
        max_width,
        max_height,
        line_spacing=4,
    ):
        fitting_font = self.get_fitting_font(
            text,
            max_width,
            max_height,
            line_spacing,
        )
        lines = text.split("\n")
        line_height = self.small_font.get_linesize()
        total_height = (line_height + line_spacing) * len(lines) - line_spacing
        top = center[1] - total_height // 2

        for line in lines:
            line_surf = fitting_font.render(line, True, WHITE)
            line_rect = line_surf.get_rect(
                center=(center[0], top + line_height // 2),
            )
            self.screen.blit(line_surf, line_rect)
            top += line_height + line_spacing

    def get_fitting_font(self, text, max_width, max_height, line_spacing):
        font_size = 10
        fitting_font = self.small_font
        lines = text.split("\n")
        line_count = len(lines)

        while True:
            test_font = pygame.font.Font(None, font_size)
            line_height = test_font.get_linesize()
            total_height = (
                line_height + line_spacing
            ) * line_count - line_spacing

            fits_width = all(
                test_font.size(line)[0] <= max_width for line in lines
            )
            fits_height = total_height <= max_height

            if fits_width and fits_height:
                fitting_font = test_font
                font_size += 1
            else:
                break

        return fitting_font

    def handle_input_box_click(self, event):
        if self.input_box.collidepoint(event.pos):
            self.active = not self.active
        else:
            self.active = False
        self.color = self.color_active if self.active else self.color_inactive

    def handle_square_click(self, mouse_pos):
        for entry in self.displayed_entries:
            for key, value_dict in entry.items():
                if value_dict["rect"] and value_dict["rect"].collidepoint(
                    mouse_pos
                ):
                    result_cycle = list(self.RESULT_MAP[key].keys())
                    if value_dict["result"] is not None:
                        result_cycle.remove(None)
                    current_result_index = result_cycle.index(
                        value_dict["result"]
                    )
                    next_result = result_cycle[
                        (current_result_index + 1) % len(result_cycle)
                    ]
                    value_dict["result"] = next_result
                    break

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_square_click(event.pos)
                self.handle_input_box_click(event)
            if event.type == pygame.KEYDOWN and self.active:
                if event.key == pygame.K_RETURN:
                    row = (
                        self.solver.df.filter(pl.col("name") == self.text)
                        .to_pandas()
                        .iloc[0]
                        .to_dict()
                    )
                    row = {
                        k: "\n".join(v) if isinstance(v, np.ndarray) else v
                        for k, v in row.items()
                    }
                    row = {
                        k: {"value": v, "result": None, "rect": None}
                        for k, v in row.items()
                    }
                    self.displayed_entries.append(row)
                    self.n_possib_entries.append(self.n_possib)
                    print(self.text)
                    self.text = ""
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
            elif (
                (not self.active)
                and event.type == pygame.KEYDOWN
                and event.key == pygame.K_RETURN
            ):
                outcome = [
                    v["result"]
                    for k, v in self.displayed_entries[-1].items()
                    if k != "name"
                ]
                self.solver.update_df_with_guess(
                    self.displayed_entries[-1]["name"]["value"],
                    outcome,
                )
                self.n_possib = len(self.solver.df)
                self.guesses = self.solver.get_best_guesses()

    def main_loop(self):
        running = True
        while running:
            self.screen.fill(BLACK)
            self.handle_events()

            self.draw_left_header()
            self.draw_right_header()
            self.draw_table()

            self.draw_text_input_box()
            self.draw_entries()

            pygame.display.flip()
            self.clock.tick(30)


if __name__ == "__main__":
    simulator = Simulator()
    simulator.main_loop()
