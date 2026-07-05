"""Pygame dashboard: stylized top-down city intersection with LLM-driven cars.

A vector-art redesign of the traffic-negotiation visualizer, styled after a
modern analytics dashboard: top bar (branding + scenario picker), a main
city-block view (roads, sidewalks, crosswalks, buildings, trees, signal
poles), a right-hand telemetry sidebar (stat cards, per-car status, a live
decision log), and a bottom playback bar (pause/step/step-back/reset, speed
control, log export).

LLM calls run on a background thread so the window stays responsive while
agents "think"; the last N world-states are kept so turns can be stepped
backward for review.
"""
from __future__ import annotations

import copy
import json
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pygame

from traffic.environment import GridIntersection, SCENARIOS, State, Vehicle, _DIRECTION_VECTOR
from traffic.agents.react_agent import choose_move

# ── layout ───────────────────────────────────────────────────────────────────
# Sized to comfortably fit inside a 1280x720 desktop with room left over for
# the OS title bar and taskbar (pygame windows are not resizable here, so the
# board/sidebar/bars must all fit within one fixed, conservative budget).
GRID_SIZE = 10
CELL = 44
MARGIN = 30
BOARD_W = BOARD_H = GRID_SIZE * CELL + MARGIN * 2
SIDEBAR = 360
TOPBAR = 56
BOTTOMBAR = 56
WIN_W = BOARD_W + SIDEBAR
WIN_H = BOARD_H + TOPBAR + BOTTOMBAR
FPS = 60
BASE_MOVE_SECONDS = 0.9
MAX_HISTORY = 200

CAR_ID_LABEL = {"N": "Car A", "S": "Car B", "E": "Car C", "W": "Car D"}
CAR_TURN_LABEL = {"STRAIGHT": "STRAIGHT", "LEFT": "LEFT", "RIGHT": "RIGHT"}

# ── palette ──────────────────────────────────────────────────────────────────
BG_APP        = (10, 12, 18)
TOPBAR_BG     = (17, 19, 27)
BOTTOMBAR_BG  = (17, 19, 27)
SIDEBAR_BG    = (15, 17, 24)
CARD_BG       = (22, 25, 34)
CARD_BG_2     = (26, 30, 41)
CARD_BORDER   = (38, 42, 56)
TEXT          = (232, 235, 242)
SUBTEXT       = (142, 148, 165)
MUTED         = (92, 97, 114)
ACCENT        = (86, 134, 251)
ACCENT_SOFT   = (40, 52, 82)

ROAD          = (52, 55, 63)
ROAD_DARK     = (44, 47, 54)
SIDEWALK      = (150, 145, 132)
SIDEWALK_EDGE = (120, 116, 104)
CURB          = (196, 190, 172)
BLOCK_LIGHT   = (99, 108, 92)
BLOCK_DARK    = (70, 78, 66)
BUILDING_1    = (58, 66, 78)
BUILDING_2    = (48, 55, 66)
ROOF_LINE     = (30, 35, 44)
TREE_DARK     = (36, 74, 46)
TREE_LIGHT    = (52, 102, 62)
LANE_LINE     = (235, 190, 70)
CROSSWALK     = (232, 233, 236)

STATUS_GREEN  = (58, 199, 129)
STATUS_AMBER  = (240, 176, 58)
STATUS_RED    = (233, 78, 92)
STATUS_BLUE   = (86, 134, 251)

CAR_COLORS = {
    "N": (66, 133, 244),   # blue   -> Car A
    "S": (52, 199, 129),   # green  -> Car B
    "E": (240, 176, 58),   # amber  -> Car C
    "W": (233, 78, 92),    # red    -> Car D
}
CAR_LABEL_BG = CAR_COLORS

ROAD_HALF_W = CELL * 1.55
LANE_OFFSET = CELL * 0.24


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))


def grid_to_px(cell_xy: tuple[float, float], heading: tuple[float, float] = (0, 0)) -> tuple[float, float]:
    gx, gy = cell_xy
    px = MARGIN + (gx + 0.5) * CELL
    py = MARGIN + (gy + 0.5) * CELL
    hx, hy = heading
    if hx > 0:
        py += LANE_OFFSET
    elif hx < 0:
        py -= LANE_OFFSET
    elif hy > 0:
        px -= LANE_OFFSET
    elif hy < 0:
        px += LANE_OFFSET
    return (px, py)


def initial_heading(v: Vehicle) -> tuple[int, int]:
    return _DIRECTION_VECTOR.get(v.direction, (1, 0))


# ── static city-block background, drawn once to a surface ──────────────────
def build_city_surface(seed: int = 7) -> pygame.Surface:
    import random
    rng = random.Random(seed)

    surf = pygame.Surface((BOARD_W, BOARD_H))
    surf.fill(BLOCK_LIGHT)

    cx = MARGIN + GRID_SIZE * CELL / 2
    cy = MARGIN + GRID_SIZE * CELL / 2

    # four city blocks (corners), each with a scattering of buildings + trees
    block_defs = [
        (int(v0), int(v1), int(v2), int(v3))
        for (v0, v1, v2, v3) in [
            (0, 0, cx - ROAD_HALF_W, cy - ROAD_HALF_W),
            (cx + ROAD_HALF_W, 0, BOARD_W, cy - ROAD_HALF_W),
            (0, cy + ROAD_HALF_W, cx - ROAD_HALF_W, BOARD_H),
            (cx + ROAD_HALF_W, cy + ROAD_HALF_W, BOARD_W, BOARD_H),
        ]
    ]
    for (x0, y0, x1, y1) in block_defs:
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        for row in range(int(h)):
            t = row / max(1, h - 1)
            color = lerp_color(BLOCK_LIGHT, BLOCK_DARK, t)
            pygame.draw.line(surf, color, (x0, y0 + row), (x1, y0 + row))

        # scatter a few buildings (rounded rects with a roof-line) and trees
        n_buildings = rng.randint(2, 3)
        for _ in range(n_buildings):
            bw = rng.randint(int(w * 0.22), int(w * 0.34))
            bh = rng.randint(int(h * 0.22), int(h * 0.34))
            bx = x0 + rng.randint(10, max(11, int(w - bw - 10)))
            by = y0 + rng.randint(10, max(11, int(h - bh - 10)))
            shade = BUILDING_1 if rng.random() > 0.5 else BUILDING_2
            shadow_rect = pygame.Rect(bx + 4, by + 5, bw, bh)
            pygame.draw.rect(surf, (0, 0, 0), shadow_rect, border_radius=6)
            rect = pygame.Rect(bx, by, bw, bh)
            pygame.draw.rect(surf, shade, rect, border_radius=6)
            pygame.draw.rect(surf, ROOF_LINE, rect, 2, border_radius=6)
            # a few window dots
            for wy in range(by + 8, by + bh - 8, 10):
                for wx in range(bx + 8, bx + bw - 8, 10):
                    pygame.draw.rect(surf, (150, 165, 190), (wx, wy, 3, 4))

        n_trees = rng.randint(3, 5)
        for _ in range(n_trees):
            tx = x0 + rng.randint(12, max(13, int(w - 12)))
            ty = y0 + rng.randint(12, max(13, int(h - 12)))
            r = rng.randint(7, 11)
            pygame.draw.circle(surf, (0, 0, 0, 40), (tx + 2, ty + 3), r)
            pygame.draw.circle(surf, rng.choice((TREE_DARK, TREE_LIGHT)), (tx, ty), r)

    # sidewalks framing the road bands
    sw = 16
    pygame.draw.rect(surf, SIDEWALK, (0, cy - ROAD_HALF_W - sw, BOARD_W, sw))
    pygame.draw.rect(surf, SIDEWALK, (0, cy + ROAD_HALF_W, BOARD_W, sw))
    pygame.draw.rect(surf, SIDEWALK, (cx - ROAD_HALF_W - sw, 0, sw, BOARD_H))
    pygame.draw.rect(surf, SIDEWALK, (cx + ROAD_HALF_W, 0, sw, BOARD_H))
    pygame.draw.rect(surf, SIDEWALK_EDGE, (0, cy - ROAD_HALF_W - sw, BOARD_W, 2))
    pygame.draw.rect(surf, SIDEWALK_EDGE, (0, cy + ROAD_HALF_W + sw - 2, BOARD_W, 2))
    pygame.draw.rect(surf, SIDEWALK_EDGE, (cx - ROAD_HALF_W - sw, 0, 2, BOARD_H))
    pygame.draw.rect(surf, SIDEWALK_EDGE, (cx + ROAD_HALF_W + sw - 2, 0, 2, BOARD_H))

    # road shadow for depth
    shadow = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
    pad = 6
    pygame.draw.rect(shadow, (0, 0, 0, 70), (0, cy - ROAD_HALF_W + pad, BOARD_W, ROAD_HALF_W * 2))
    pygame.draw.rect(shadow, (0, 0, 0, 70), (cx - ROAD_HALF_W + pad, 0, ROAD_HALF_W * 2, BOARD_H))
    surf.blit(shadow, (2, 2))

    pygame.draw.rect(surf, ROAD, (0, cy - ROAD_HALF_W, BOARD_W, ROAD_HALF_W * 2))
    pygame.draw.rect(surf, ROAD, (cx - ROAD_HALF_W, 0, ROAD_HALF_W * 2, BOARD_H))
    pygame.draw.rect(surf, ROAD_DARK,
                      (cx - ROAD_HALF_W, cy - ROAD_HALF_W, ROAD_HALF_W * 2, ROAD_HALF_W * 2))
    pygame.draw.rect(surf, CURB, (0, cy - ROAD_HALF_W, BOARD_W, ROAD_HALF_W * 2), 2)
    pygame.draw.rect(surf, CURB, (cx - ROAD_HALF_W, 0, ROAD_HALF_W * 2, BOARD_H), 2)

    dash_len, gap_len = 18, 14

    def dashed_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length == 0:
            return
        dx, dy = (x2 - x1) / length, (y2 - y1) / length
        dist = 0.0
        while dist < length:
            seg_start = (x1 + dx * dist, y1 + dy * dist)
            seg_end_d = min(dist + dash_len, length)
            seg_end = (x1 + dx * seg_end_d, y1 + dy * seg_end_d)
            mx, my = (seg_start[0] + seg_end[0]) / 2, (seg_start[1] + seg_end[1]) / 2
            if not (cx - ROAD_HALF_W < mx < cx + ROAD_HALF_W and cy - ROAD_HALF_W < my < cy + ROAD_HALF_W):
                pygame.draw.line(surf, LANE_LINE, seg_start, seg_end, 3)
            dist += dash_len + gap_len

    dashed_line((0, cy), (BOARD_W, cy))
    dashed_line((cx, 0), (cx, BOARD_H))

    stripe_w, stripe_gap, n_stripes = 7, 6, 12

    def crosswalk(vertical: bool, near_edge: float):
        if vertical:
            x0 = cx - ROAD_HALF_W
            for i in range(n_stripes):
                sx = x0 + i * (stripe_w + stripe_gap)
                if sx > cx + ROAD_HALF_W:
                    break
                pygame.draw.rect(surf, CROSSWALK, (sx, near_edge, stripe_w, 16))
        else:
            y0 = cy - ROAD_HALF_W
            for i in range(n_stripes):
                sy = y0 + i * (stripe_w + stripe_gap)
                if sy > cy + ROAD_HALF_W:
                    break
                pygame.draw.rect(surf, CROSSWALK, (near_edge, sy, 16, stripe_w))

    crosswalk(True, cy - ROAD_HALF_W - 20)
    crosswalk(True, cy + ROAD_HALF_W + 4)
    crosswalk(False, cx - ROAD_HALF_W - 20)
    crosswalk(False, cx + ROAD_HALF_W + 4)

    # signal poles at each corner of the intersection
    for (px, py) in [
        (cx - ROAD_HALF_W - 8, cy - ROAD_HALF_W - 8),
        (cx + ROAD_HALF_W + 8, cy - ROAD_HALF_W - 8),
        (cx - ROAD_HALF_W - 8, cy + ROAD_HALF_W + 8),
        (cx + ROAD_HALF_W + 8, cy + ROAD_HALF_W + 8),
    ]:
        pygame.draw.circle(surf, (70, 74, 84), (int(px), int(py)), 4)
        pygame.draw.line(surf, (70, 74, 84), (px, py), (px, py - 26), 3)

    return surf


def draw_traffic_lights(surf: pygame.Surface, env: GridIntersection):
    cx = MARGIN + GRID_SIZE * CELL / 2
    cy = MARGIN + GRID_SIZE * CELL / 2
    offset = ROAD_HALF_W + 22
    nudge = 15

    positions = {
        "N": (cx - nudge, cy - offset),
        "S": (cx + nudge, cy + offset),
        "E": (cx + offset, cy - nudge),
        "W": (cx - offset, cy + nudge),
    }

    for direction, (px, py) in positions.items():
        phase = env.light_for(direction)
        w, h = 13, 27
        housing = pygame.Rect(0, 0, w, h)
        housing.center = (px, py)
        pygame.draw.rect(surf, (24, 25, 30), housing, border_radius=5)
        pygame.draw.rect(surf, (8, 8, 10), housing, 2, border_radius=5)

        is_all_red = env.state.light_phase == "ALL_RED"
        r_color = STATUS_RED if phase == "RED" else (55, 22, 26)
        y_color = STATUS_AMBER if is_all_red else (55, 46, 22)
        g_color = STATUS_GREEN if phase == "GREEN" else (18, 44, 30)

        cyc = h / 3
        for i, color in enumerate((r_color, y_color, g_color)):
            cy_dot = housing.top + cyc * (i + 0.5)
            if color in (STATUS_RED, STATUS_AMBER, STATUS_GREEN):
                glow = pygame.Surface((24, 24), pygame.SRCALPHA)
                pygame.draw.circle(glow, (*color, 90), (12, 12), 10)
                surf.blit(glow, (px - 12, cy_dot - 12))
            pygame.draw.circle(surf, color, (px, cy_dot), 5)


def draw_goal_marker(surf: pygame.Surface, pos: tuple[int, int], color: tuple[int, int, int], pulse_t: float):
    px, py = grid_to_px(pos)
    radius = 15 + 3 * math.sin(pulse_t * 3.0)
    pulse = pygame.Surface((42, 42), pygame.SRCALPHA)
    pygame.draw.circle(pulse, (*color, 50), (21, 21), int(radius))
    surf.blit(pulse, (px - 21, py - 21))
    pygame.draw.circle(surf, color, (int(px), int(py)), 12, 3)
    pygame.draw.circle(surf, color, (int(px), int(py)), 3)


def heading_to_angle(heading: tuple[float, float]) -> float:
    """Convert a (dx, dy) heading into a pygame rotation angle (degrees, CCW)."""
    hx, hy = heading
    if hx == 0 and hy == 0:
        return 0.0
    return -math.degrees(math.atan2(hy, hx))


def shortest_angle_lerp(a: float, b: float, t: float) -> float:
    """Interpolate between two angles (degrees) taking the shorter way around."""
    diff = (b - a + 180) % 360 - 180
    return a + diff * t


def draw_car(surf: pygame.Surface, px: float, py: float, angle: float,
             color: tuple[int, int, int], label: str, font: pygame.font.Font,
             arrived: bool, flash: bool, waiting: bool):
    w, h = 22, 13

    shadow = pygame.Surface((w + 8, h + 8), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow, (0, 0, 0, 100), shadow.get_rect())
    surf.blit(shadow, (px - (w + 8) / 2 + 2, py - (h + 8) / 2 + 3))

    car_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    body_color = (90, 230, 140) if arrived else color
    lighter = tuple(min(255, c + 45) for c in body_color)
    darker = tuple(max(0, c - 35) for c in body_color)

    for row in range(h):
        t = row / max(1, h - 1)
        shade = lerp_color(lighter, darker, t)
        pygame.draw.line(car_surf, shade, (2, row), (w - 2, row))
    pygame.draw.rect(car_surf, (14, 14, 17), (0, 0, w, h), 1, border_radius=4)

    pygame.draw.rect(car_surf, (205, 226, 244), (w - 8, 2, 5, h - 4), border_radius=1)
    pygame.draw.rect(car_surf, (255, 244, 200), (w - 2, 1, 2, 2), border_radius=1)
    pygame.draw.rect(car_surf, (255, 244, 200), (w - 2, h - 3, 2, 2), border_radius=1)
    pygame.draw.rect(car_surf, (200, 40, 40), (1, 1, 2, 2), border_radius=1)
    pygame.draw.rect(car_surf, (200, 40, 40), (1, h - 3, 2, 2), border_radius=1)

    if waiting and not arrived:
        rear = pygame.Surface((4, h), pygame.SRCALPHA)
        pygame.draw.rect(rear, (255, 60, 60, 200), (0, 1, 3, h - 2), border_radius=1)
        car_surf.blit(rear, (0, 0))

    rotated = pygame.transform.rotate(car_surf, angle)
    rect = rotated.get_rect(center=(px, py))

    if flash:
        glow = pygame.Surface((w + 16, h + 16), pygame.SRCALPHA)
        pygame.draw.ellipse(glow, (*STATUS_RED, 140), glow.get_rect())
        surf.blit(glow, glow.get_rect(center=(px, py)))

    surf.blit(rotated, rect)

    label_surf = font.render(label, True, (255, 255, 255))
    lbl_rect = label_surf.get_rect(center=(px, py - 16))
    bg = pygame.Surface((lbl_rect.w + 8, lbl_rect.h + 4), pygame.SRCALPHA)
    pygame.draw.rect(bg, (*color, 235), bg.get_rect(), border_radius=5)
    surf.blit(bg, (lbl_rect.x - 4, lbl_rect.y - 2))
    surf.blit(label_surf, lbl_rect)


# ── UI widgets ───────────────────────────────────────────────────────────────
@dataclass
class Button:
    rect: pygame.Rect
    label: str
    kind: str = "ghost"     # "primary" | "ghost"
    enabled: bool = True

    def draw(self, surf, font):
        bg = ACCENT if self.kind == "primary" else CARD_BG_2
        border = ACCENT if self.kind == "primary" else CARD_BORDER
        fg = (255, 255, 255) if self.kind == "primary" else TEXT
        if not self.enabled:
            bg = CARD_BG
            fg = MUTED
        pygame.draw.rect(surf, bg, self.rect, border_radius=8)
        pygame.draw.rect(surf, border, self.rect, 1, border_radius=8)
        txt = font.render(self.label, True, fg)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def hit(self, pos) -> bool:
        return self.enabled and self.rect.collidepoint(pos)


@dataclass
class Dropdown:
    rect: pygame.Rect
    options: list[str]
    selected: int = 0
    open_: bool = False

    @property
    def value(self) -> str:
        return self.options[self.selected]

    def draw(self, surf, font, font_small):
        pygame.draw.rect(surf, CARD_BG_2, self.rect, border_radius=8)
        pygame.draw.rect(surf, CARD_BORDER, self.rect, 1, border_radius=8)
        txt = font_small.render(self.value, True, TEXT)
        surf.blit(txt, (self.rect.x + 12, self.rect.y + (self.rect.h - txt.get_height()) // 2))
        # caret
        cx, cy = self.rect.right - 16, self.rect.centery
        pygame.draw.polygon(surf, SUBTEXT, [(cx - 5, cy - 3), (cx + 5, cy - 3), (cx, cy + 4)])

        if self.open_:
            item_h = 30
            menu_rect = pygame.Rect(self.rect.x, self.rect.bottom + 4, self.rect.w + 60, item_h * len(self.options))
            pygame.draw.rect(surf, CARD_BG_2, menu_rect, border_radius=8)
            pygame.draw.rect(surf, CARD_BORDER, menu_rect, 1, border_radius=8)
            for i, opt in enumerate(self.options):
                row = pygame.Rect(menu_rect.x, menu_rect.y + i * item_h, menu_rect.w, item_h)
                if i == self.selected:
                    pygame.draw.rect(surf, ACCENT_SOFT, row, border_radius=6)
                lbl = font_small.render(opt, True, TEXT)
                surf.blit(lbl, (row.x + 12, row.y + (item_h - lbl.get_height()) // 2))

    def menu_rect(self) -> pygame.Rect:
        item_h = 30
        return pygame.Rect(self.rect.x, self.rect.bottom + 4, self.rect.w + 60, item_h * len(self.options))

    def handle_click(self, pos) -> bool:
        """Returns True if the click was consumed by this widget."""
        if self.rect.collidepoint(pos):
            self.open_ = not self.open_
            return True
        if self.open_:
            menu = self.menu_rect()
            if menu.collidepoint(pos):
                item_h = 30
                idx = (pos[1] - menu.y) // item_h
                if 0 <= idx < len(self.options):
                    self.selected = int(idx)
                self.open_ = False
                return True
            self.open_ = False
        return False


# ── top bar ──────────────────────────────────────────────────────────────────
def draw_topbar(surf, font_bold, font_small, scenario_dropdown: Dropdown, new_scenario_btn: Button):
    pygame.draw.rect(surf, TOPBAR_BG, (0, 0, WIN_W, TOPBAR))
    pygame.draw.line(surf, CARD_BORDER, (0, TOPBAR), (WIN_W, TOPBAR), 1)

    # logo mark
    logo_rect = pygame.Rect(20, 16, 36, 36)
    pygame.draw.rect(surf, ACCENT, logo_rect, border_radius=9)
    logo_txt = font_bold.render("A", True, (255, 255, 255))
    surf.blit(logo_txt, logo_txt.get_rect(center=logo_rect.center))

    title = font_bold.render("LLM Traffic Negotiation", True, TEXT)
    surf.blit(title, (68, 14))
    sub = font_small.render("Multi-Agent Intersection Simulation", True, SUBTEXT)
    surf.blit(sub, (68, 38))

    scenario_dropdown.draw(surf, font_small, font_small)
    lbl = font_small.render("Scenario", True, SUBTEXT)
    surf.blit(lbl, (scenario_dropdown.rect.x - lbl.get_width() - 10,
                    scenario_dropdown.rect.y + (scenario_dropdown.rect.h - lbl.get_height()) // 2))

    new_scenario_btn.draw(surf, font_small)


# ── bottom playback bar ──────────────────────────────────────────────────────
def draw_bottombar(surf, font_small, buttons: dict[str, Button], speed_dropdown: Dropdown, playing: bool):
    y0 = TOPBAR + BOARD_H
    pygame.draw.rect(surf, BOTTOMBAR_BG, (0, y0, WIN_W, BOTTOMBAR))
    pygame.draw.line(surf, CARD_BORDER, (0, y0), (WIN_W, y0), 1)

    buttons["play_pause"].label = "Pause" if playing else "Play"
    for key in ("play_pause", "step", "step_back", "reset"):
        buttons[key].draw(surf, font_small)

    lbl = font_small.render("Speed", True, SUBTEXT)
    surf.blit(lbl, (speed_dropdown.rect.x - lbl.get_width() - 10,
                    speed_dropdown.rect.y + (speed_dropdown.rect.h - lbl.get_height()) // 2))
    speed_dropdown.draw(surf, font_small, font_small)

    for key in ("logs", "export"):
        buttons[key].draw(surf, font_small)


# ── sidebar ──────────────────────────────────────────────────────────────────
class Sidebar:
    def __init__(self, font, font_small, font_bold):
        self.font = font
        self.font_small = font_small
        self.font_bold = font_bold
        self.log: list[tuple[int, str, tuple[int, int, int], str]] = []  # (turn, text, color, status)

    def add(self, turn: int, text: str, color=TEXT, status: str = "ACCEPTED"):
        self.log.append((turn, text, color, status))
        self.log = self.log[-60:]

    def _card(self, surf, x, y, w, h, label, value, value_color, icon_color=None):
        pygame.draw.rect(surf, CARD_BG, (x, y, w, h), border_radius=10)
        pygame.draw.rect(surf, CARD_BORDER, (x, y, w, h), 1, border_radius=10)
        lbl = self.font_small.render(label, True, SUBTEXT)
        surf.blit(lbl, (x + 12, y + 8))
        val = self.font_bold.render(str(value), True, value_color)
        surf.blit(val, (x + 12, y + 26))

    def draw(self, surf, x0, turn, max_turns, collisions, status_summary,
              vehicles, colors, speeds_kmh, agent_states, negotiation_state, negotiation_desc):
        pygame.draw.rect(surf, SIDEBAR_BG, (x0, TOPBAR, SIDEBAR, BOARD_H))
        pygame.draw.line(surf, CARD_BORDER, (x0, TOPBAR), (x0, TOPBAR + BOARD_H), 1)

        pad = 16
        y = TOPBAR + 16

        card_w = (SIDEBAR - pad * 2 - 16) / 3
        card_h = 52
        self._card(surf, x0 + pad, y, card_w, card_h, "Current Turn", f"{turn} / {max_turns}", TEXT)
        self._card(surf, x0 + pad + card_w + 8, y, card_w, card_h, "Collisions", collisions,
                    STATUS_RED if collisions else STATUS_GREEN)
        self._card(surf, x0 + pad + 2 * (card_w + 8), y, card_w, card_h, "Overall Status", status_summary,
                    STATUS_GREEN if "arrived" in status_summary.lower() else TEXT)
        y += card_h + 18

        header = self.font_bold.render("Agent Status", True, TEXT)
        surf.blit(header, (x0 + pad, y))
        y += 26

        n = len(vehicles)
        agent_w = (SIDEBAR - pad * 2 - (n - 1) * 8) / max(n, 1)
        agent_h = 78
        for i, v in enumerate(vehicles):
            ax = x0 + pad + i * (agent_w + 8)
            color = colors.get(v.id, TEXT)
            pygame.draw.rect(surf, CARD_BG, (ax, y, agent_w, agent_h), border_radius=10)
            pygame.draw.rect(surf, color, (ax, y, agent_w, 3))
            pygame.draw.rect(surf, CARD_BORDER, (ax, y, agent_w, agent_h), 1, border_radius=10)
            name = self.font_small.render(CAR_ID_LABEL.get(v.id, v.id), True, TEXT)
            surf.blit(name, (ax + 10, y + 10))
            st = agent_states.get(v.id, "Waiting")
            st_color = {"Moving": STATUS_BLUE, "Waiting": SUBTEXT, "Yielding": STATUS_AMBER,
                        "Arrived": STATUS_GREEN}.get(st, TEXT)
            pygame.draw.circle(surf, st_color, (ax + 12, y + 34), 4)
            st_txt = self.font_small.render(st, True, st_color)
            surf.blit(st_txt, (ax + 22, y + 27))
            speed_txt = self.font_small.render(f"{speeds_kmh.get(v.id, 0)} km/h", True, SUBTEXT)
            surf.blit(speed_txt, (ax + 10, y + 50))
        y += agent_h + 18

        pygame.draw.rect(surf, CARD_BG, (x0 + pad, y, SIDEBAR - pad * 2, 62), border_radius=10)
        pygame.draw.rect(surf, CARD_BORDER, (x0 + pad, y, SIDEBAR - pad * 2, 62), 1, border_radius=10)
        neg_title = self.font_bold.render("Negotiation State", True, TEXT)
        surf.blit(neg_title, (x0 + pad + 12, y + 8))
        pill_color = {"Normal Flow": STATUS_GREEN, "Yielding": STATUS_AMBER,
                      "Conflict": STATUS_RED}.get(negotiation_state, STATUS_GREEN)
        pill_txt = self.font_small.render(negotiation_state, True, (15, 15, 18))
        pill_w = pill_txt.get_width() + 20
        pill_rect = pygame.Rect(x0 + SIDEBAR - pad - pill_w - 12, y + 7, pill_w, 22)
        pygame.draw.rect(surf, pill_color, pill_rect, border_radius=11)
        surf.blit(pill_txt, (pill_rect.x + 10, pill_rect.y + 3))
        desc_lines = self._wrap(negotiation_desc, SIDEBAR - pad * 2 - 24)
        dy = y + 32
        for line in desc_lines[:2]:
            d = self.font_small.render(line, True, SUBTEXT)
            surf.blit(d, (x0 + pad + 12, dy))
            dy += 16
        y += 62 + 18

        row_header = self.font_bold.render("Recent Agent Decisions", True, TEXT)
        surf.blit(row_header, (x0 + pad, y))
        live_txt = self.font_small.render("● Live", True, STATUS_GREEN)
        surf.blit(live_txt, (x0 + SIDEBAR - pad - live_txt.get_width(), y + 2))
        y += 26

        max_y = TOPBAR + BOARD_H - 10
        for turn_no, text, color, log_status in reversed(self.log):
            if y > max_y:
                break
            row_h = 20
            turn_txt = self.font_small.render(f"Turn {turn_no}", True, MUTED)
            surf.blit(turn_txt, (x0 + pad, y))
            body_txt = self.font_small.render(text, True, color)
            surf.blit(body_txt, (x0 + pad + 62, y))
            status_color = STATUS_GREEN if log_status == "ACCEPTED" else STATUS_AMBER
            status_txt = self.font_small.render(log_status, True, status_color)
            surf.blit(status_txt, (x0 + SIDEBAR - pad - status_txt.get_width(), y))
            y += row_h

    def _wrap(self, text, max_w):
        words = text.split(" ")
        lines, cur = [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if self.font_small.size(trial)[0] > max_w and cur:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines


# ── background LLM worker ───────────────────────────────────────────────────
class TurnWorker(threading.Thread):
    def __init__(self, env: GridIntersection, vehicle_ids: list[str], out_q: "queue.Queue"):
        super().__init__(daemon=True)
        self.env = env
        self.vehicle_ids = vehicle_ids
        self.out_q = out_q

    def run(self):
        moves: dict[str, tuple[int, int]] = {}
        for vid in self.vehicle_ids:
            try:
                move = choose_move(self.env.state, vid, self.env)
            except Exception as exc:  # pragma: no cover - safety net for live demo
                move = (0, 0)
                self.out_q.put(("log", (f"{CAR_ID_LABEL.get(vid, vid)}: LLM error ({exc}) -> WAIT", STATUS_RED)))
            moves[vid] = move
        self.out_q.put(("moves", moves))


_MOVE_TOKEN = {
    (1, 0): "RIGHT", (-1, 0): "LEFT", (0, 1): "DOWN", (0, -1): "UP", (0, 0): "WAIT",
    (2, 0): "RIGHT2", (-2, 0): "LEFT2", (0, 2): "DOWN2", (0, -2): "UP2",
}


def _move_label(v: Vehicle, mv: tuple[int, int]) -> str:
    """Describe the move relative to the car's own turn intent when possible."""
    if mv == (0, 0):
        return "WAIT"
    if mv == initial_heading(v) or mv == tuple(2 * x for x in initial_heading(v)):
        return v.turn if v.speed or True else "STRAIGHT"
    return _MOVE_TOKEN.get(mv, str(mv))


@dataclass
class Snapshot:
    """A fully independent copy of one point in the episode, for step-back."""
    state: State
    env_collisions: int
    env_near_misses: int
    env_violations: int
    light_phase: str
    phase_timer: int


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("LLM Traffic Negotiation — Intersection Dashboard")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Segoe UI", 15)
        self.font_small = pygame.font.SysFont("Segoe UI", 12)
        self.font_bold = pygame.font.SysFont("Segoe UI", 16, bold=True)
        self.font_title = pygame.font.SysFont("Segoe UI", 18, bold=True)

        self.city_surface = build_city_surface()
        self.sidebar = Sidebar(self.font, self.font_small, self.font_bold)

        self.scenario_names = list(SCENARIOS.keys())
        self.scenario_dropdown = Dropdown(pygame.Rect(WIN_W - 300, 20, 130, 30), self.scenario_names)
        self.new_scenario_btn = Button(pygame.Rect(WIN_W - 150, 20, 130, 30), "+ New Scenario")
        self.settings_btn = Button(pygame.Rect(WIN_W - 44, 20, 30, 30), "⚙")

        by0 = TOPBAR + BOARD_H + 14
        self.buttons = {
            "play_pause": Button(pygame.Rect(20, by0, 90, 36), "Pause", kind="primary"),
            "step": Button(pygame.Rect(118, by0, 70, 36), "Step"),
            "step_back": Button(pygame.Rect(196, by0, 96, 36), "Step Back"),
            "reset": Button(pygame.Rect(300, by0, 70, 36), "Reset"),
            "logs": Button(pygame.Rect(WIN_W - 190, by0, 70, 36), "Logs"),
            "export": Button(pygame.Rect(WIN_W - 112, by0, 92, 36), "Export"),
        }
        self.speed_dropdown = Dropdown(pygame.Rect(430, by0 + 3, 70, 30), ["0.5x", "1.0x", "2.0x", "4.0x"], selected=1)

        self.history: list[Snapshot] = []
        self.history_cursor = -1  # index into history when stepped back; -1 = live
        self.export_message = ""
        self.export_message_t = 0.0

        self._load_scenario(self.scenario_names[0])

    # ── scenario / episode lifecycle ────────────────────────────────────
    def _load_scenario(self, name: str):
        self.env = GridIntersection(max_turns=40, scenario=name)
        self.state = self.env.reset()
        self.vehicle_order = [v.id for v in self.state.vehicles]
        self.colors = {vid: CAR_COLORS.get(vid, (200, 200, 200)) for vid in self.vehicle_order}
        self.anim_from = {v.id: v.pos for v in self.state.vehicles}
        self.anim_to = {v.id: v.pos for v in self.state.vehicles}
        self.heading = {v.id: initial_heading(v) for v in self.state.vehicles}
        self.anim_from_heading = dict(self.heading)
        self.anim_to_heading = dict(self.heading)
        self.agent_states = {v.id: "Waiting" for v in self.state.vehicles}
        self.negotiation_state = "Normal Flow"
        self.negotiation_desc = "Agents are negotiating right-of-way using natural language and traffic rules. No conflicts detected."

        self.sidebar.log = []
        self.sidebar.add(0, f"Scenario loaded: {name}.", SUBTEXT, "INFO")
        for v in self.state.vehicles:
            self.sidebar.add(0, f"{CAR_ID_LABEL[v.id]} ({v.direction}) intends {v.turn}.", self.colors[v.id], "INFO")

        self.phase = "idle"
        self.playing = True
        self.out_q: "queue.Queue" = queue.Queue()
        self.worker: TurnWorker | None = None
        self.anim_start_t = 0.0
        self.pending_moves: dict[str, tuple[int, int]] = {}
        self.collision_flash: set[tuple[int, int]] = set()
        self.status_text = "Starting..."

        self.history = [self._snapshot()]
        self.history_cursor = -1

    def _snapshot(self) -> Snapshot:
        return Snapshot(
            state=copy.deepcopy(self.state),
            env_collisions=self.env.collisions,
            env_near_misses=self.env.near_misses,
            env_violations=self.env.red_light_violations,
            light_phase=self.env.state.light_phase,
            phase_timer=self.env.state.phase_timer,
        )

    def _push_history(self):
        self.history.append(self._snapshot())
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

    def _restore_snapshot(self, snap: Snapshot):
        self.state = copy.deepcopy(snap.state)
        self.env.state = self.state
        self.env.collisions = snap.env_collisions
        self.env.near_misses = snap.env_near_misses
        self.env.red_light_violations = snap.env_violations
        for v in self.state.vehicles:
            self.anim_from[v.id] = v.pos
            self.anim_to[v.id] = v.pos
            if v.last_move != (0, 0):
                self.heading[v.id] = (
                    1 if v.last_move[0] > 0 else -1 if v.last_move[0] < 0 else 0,
                    1 if v.last_move[1] > 0 else -1 if v.last_move[1] < 0 else 0,
                )
            self.anim_from_heading[v.id] = self.heading[v.id]
            self.anim_to_heading[v.id] = self.heading[v.id]
        self.collision_flash = set()
        self.phase = "idle"
        self.status_text = "Stepped back."

    # ── playback controls ────────────────────────────────────────────────
    def step_back(self):
        if self.history_cursor == -1:
            self.history_cursor = len(self.history) - 1
        if self.history_cursor > 0:
            self.history_cursor -= 1
            self._restore_snapshot(self.history[self.history_cursor])
            self.playing = False

    def step_forward_one(self):
        """Manual single-step: replays from history if we've stepped back, else advances live."""
        if self.history_cursor != -1 and self.history_cursor < len(self.history) - 1:
            self.history_cursor += 1
            self._restore_snapshot(self.history[self.history_cursor])
            return
        self.history_cursor = -1
        self._advance_one_turn(blocking=True)

    def reset(self):
        current = self.scenario_dropdown.value
        self._load_scenario(current)

    def export_logs(self):
        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = out_dir / f"gui_log_{stamp}.json"
        rows = [{"turn": t, "text": txt, "status": st} for (t, txt, _c, st) in self.sidebar.log]
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        self.export_message = f"Exported {len(rows)} entries -> {path}"
        self.export_message_t = time.time()

    # ── simulation step (live, LLM-backed) ──────────────────────────────
    def _advance_one_turn(self, blocking: bool = False):
        if self.env.is_done():
            return
        active_ids = [v.id for v in self.state.vehicles if not v.arrived]
        worker = TurnWorker(self.env, active_ids, queue.Queue())
        worker.start()
        worker.join()
        try:
            while True:
                kind, payload = worker.out_q.get_nowait()
                if kind == "moves":
                    self._apply_moves(payload)
        except queue.Empty:
            pass

    def _apply_moves(self, moves: dict[str, tuple[int, int]]):
        prev_positions = {v.id: v.pos for v in self.state.vehicles}
        prev_collisions = self.env.collisions
        prev_violations = self.env.red_light_violations
        prev_heading = dict(self.heading)
        for vid, mv in moves.items():
            v = next(x for x in self.state.vehicles if x.id == vid)
            label = _move_label(v, mv)
            self.sidebar.add(self.state.turn, f"{CAR_ID_LABEL.get(vid, vid)} -> {label}", self.colors.get(vid, TEXT))
            if mv != (0, 0):
                self.heading[vid] = (1 if mv[0] > 0 else -1 if mv[0] < 0 else 0,
                                      1 if mv[1] > 0 else -1 if mv[1] < 0 else 0)
            self.agent_states[vid] = "Moving" if mv != (0, 0) else "Waiting"

        self.state = self.env.step(moves)

        self.collision_flash = set()
        if self.env.collisions > prev_collisions:
            for v in self.state.vehicles:
                if moves.get(v.id, (0, 0)) != (0, 0) and v.pos == prev_positions.get(v.id):
                    self.collision_flash.add(v.pos)
            self.sidebar.add(self.state.turn, "Collision detected — cars held position.", STATUS_RED, "CONFLICT")
            self.negotiation_state = "Conflict"
            self.negotiation_desc = "A collision was detected this turn; involved agents held their prior position."
        elif self.env.red_light_violations > prev_violations:
            self.negotiation_state = "Yielding"
            self.negotiation_desc = "A car was stopped short of the box to respect the red light."
        else:
            self.negotiation_state = "Normal Flow"
            self.negotiation_desc = "Agents are negotiating right-of-way using natural language and traffic rules. No conflicts detected."

        for v in self.state.vehicles:
            self.anim_from[v.id] = prev_positions[v.id]
            self.anim_to[v.id] = v.pos
            self.anim_from_heading[v.id] = prev_heading.get(v.id, self.heading[v.id])
            self.anim_to_heading[v.id] = self.heading[v.id]
            if v.arrived:
                self.agent_states[v.id] = "Arrived"

        self.anim_start_t = time.time()
        self.status_text = "Moving..."
        self.phase = "animating"
        self._push_history()
        self.history_cursor = -1

    # ── main loop ────────────────────────────────────────────────────────
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.playing = not self.playing
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_click(event.pos)
        return True

    def _handle_click(self, pos):
        if self.scenario_dropdown.handle_click(pos):
            if self.scenario_dropdown.value != getattr(self, "_active_scenario", None):
                self._active_scenario = self.scenario_dropdown.value
                self._load_scenario(self.scenario_dropdown.value)
            return
        if self.speed_dropdown.handle_click(pos):
            return
        if self.new_scenario_btn.hit(pos):
            idx = (self.scenario_dropdown.selected + 1) % len(self.scenario_names)
            self.scenario_dropdown.selected = idx
            self._load_scenario(self.scenario_names[idx])
            return
        if self.buttons["play_pause"].hit(pos):
            self.playing = not self.playing
            return
        if self.buttons["step"].hit(pos):
            self.playing = False
            self.step_forward_one()
            return
        if self.buttons["step_back"].hit(pos):
            self.step_back()
            return
        if self.buttons["reset"].hit(pos):
            self.reset()
            return
        if self.buttons["export"].hit(pos):
            self.export_logs()
            return
        # "Logs" is a no-op toggle placeholder (the decision log is always visible in the sidebar)

    @property
    def speed_multiplier(self) -> float:
        return {"0.5x": 0.5, "1.0x": 1.0, "2.0x": 2.0, "4.0x": 4.0}[self.speed_dropdown.value]

    def update(self):
        now = time.time()
        move_seconds = BASE_MOVE_SECONDS / self.speed_multiplier

        if not self.playing or self.history_cursor != -1:
            return

        if self.phase == "idle":
            if self.env.is_done():
                self.phase = "done"
                self.status_text = "SUCCESS — all agents arrived!" if self.env.all_arrived() else "Time limit reached."
                self.sidebar.add(self.state.turn, self.status_text,
                                  STATUS_GREEN if self.env.all_arrived() else STATUS_RED, "INFO")
                self.playing = False
            else:
                active_ids = [v.id for v in self.state.vehicles if not v.arrived]
                self.status_text = "Agents thinking..."
                self.worker = TurnWorker(self.env, active_ids, self.out_q)
                self.worker.start()
                self.phase = "thinking"

        elif self.phase == "thinking":
            try:
                while True:
                    kind, payload = self.out_q.get_nowait()
                    if kind == "log":
                        text, color = payload
                        self.sidebar.add(self.state.turn, text, color, "INFO")
                    elif kind == "moves":
                        self.pending_moves = payload
            except queue.Empty:
                pass

            if self.worker is not None and not self.worker.is_alive() and self.pending_moves:
                moves, self.pending_moves = self.pending_moves, {}
                self._apply_moves(moves)

        elif self.phase == "animating":
            if now - self.anim_start_t >= move_seconds:
                self.phase = "idle"
                self.status_text = "Resolved."

    def _status_summary(self) -> str:
        if self.env.all_arrived():
            return "All agents arrived"
        if self.env.is_done():
            return "Time limit reached"
        return "In progress"

    def draw(self):
        now = time.time()
        move_seconds = BASE_MOVE_SECONDS / self.speed_multiplier
        self.screen.fill(BG_APP)

        board = pygame.Surface((BOARD_W, BOARD_H))
        board.blit(self.city_surface, (0, 0))

        for v in self.state.vehicles:
            draw_goal_marker(board, v.goal, self.colors.get(v.id, (255, 255, 255)), now)

        draw_traffic_lights(board, self.env)

        t = min(1.0, (now - self.anim_start_t) / move_seconds) if self.phase == "animating" else 1.0
        ease = t * t * (3 - 2 * t)

        for v in self.state.vehicles:
            from_h = self.anim_from_heading.get(v.id, self.heading.get(v.id, (1, 0)))
            to_h = self.anim_to_heading.get(v.id, self.heading.get(v.id, (1, 0)))
            fx, fy = self.anim_from.get(v.id, v.pos)
            tx, ty = self.anim_to.get(v.id, v.pos)

            # position: straight-line interpolation in grid space, each
            # endpoint using its own heading's lane offset so the pivot cell
            # doesn't jump sideways when the heading changes mid-turn
            start_px, start_py = grid_to_px((fx, fy), from_h)
            end_px, end_py = grid_to_px((tx, ty), to_h)
            px = start_px + (end_px - start_px) * ease
            py = start_py + (end_py - start_py) * ease

            # the sprite always faces the direction it is actually travelling
            # this turn (the straight line from start to end), so it never
            # looks like it's sliding sideways while pointed the wrong way
            dx, dy = end_px - start_px, end_py - start_py
            if dx == 0 and dy == 0:
                angle = heading_to_angle(to_h if to_h != (0, 0) else from_h)
            else:
                angle = heading_to_angle((dx, dy))

            flash = v.pos in self.collision_flash
            waiting = v.speed == 0 and not v.arrived and self.phase == "idle"
            draw_car(board, px, py, angle, self.colors.get(v.id, (200, 200, 200)),
                      CAR_ID_LABEL.get(v.id, v.id), self.font_small, v.arrived, flash, waiting)

        self.screen.blit(board, (0, TOPBAR))

        draw_topbar(self.screen, self.font_title, self.font_small, self.scenario_dropdown, self.new_scenario_btn)
        self.settings_btn.draw(self.screen, self.font_small)

        speeds_kmh = {v.id: v.speed * 18 for v in self.state.vehicles}
        self.sidebar.draw(
            self.screen, BOARD_W, self.state.turn, self.env.max_turns, self.env.collisions,
            self._status_summary(), self.state.vehicles, self.colors, speeds_kmh,
            self.agent_states, self.negotiation_state, self.negotiation_desc,
        )

        draw_bottombar(self.screen, self.font_small, self.buttons, self.speed_dropdown, self.playing)

        if self.export_message and now - self.export_message_t < 3.0:
            msg = self.font_small.render(self.export_message, True, STATUS_GREEN)
            self.screen.blit(msg, (20, TOPBAR + BOARD_H - 22))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)
            running = self.handle_events()
            self.update()
            self.draw()
        pygame.quit()
        sys.exit()


def main():
    App().run()


if __name__ == "__main__":
    main()
