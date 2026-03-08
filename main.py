import time
import tkinter
import turtle
from turtle import Screen, Turtle

from snake_env import SnakeEnv, UP, DOWN, LEFT, RIGHT, GRID_SIZE
from score import Scoreboard

CELL_SIZE = 40
GRID_CENTER = GRID_SIZE // 2

BASE_DELAY = 0.15
MIN_DELAY = 0.05
SPEED_STEP = 0.01
FOOD_PER_SPEEDUP = 3


def grid_to_pixel(row, col):
    x = (col - GRID_CENTER) * CELL_SIZE
    y = (GRID_CENTER - row) * CELL_SIZE
    return x, y


def get_step_delay(score):
    reductions = score // FOOD_PER_SPEEDUP
    return max(MIN_DELAY, BASE_DELAY - reductions * SPEED_STEP)


# ── Screen setup ──────────────────────────────────────────
WINDOW_W = GRID_SIZE * CELL_SIZE + 20
WINDOW_H = GRID_SIZE * CELL_SIZE + 80
screen = Screen()
screen.setup(width=WINDOW_W, height=WINDOW_H)
screen.bgcolor("black")
screen.title("Funky Snake Game")
screen.tracer(0)

# ── Draw visible border around the grid ──────────────────
border = Turtle()
border.hideturtle()
border.penup()
border.color("grey")
border.pensize(2)
half = GRID_SIZE * CELL_SIZE / 2
border.goto(-half, half)
border.pendown()
for _ in range(4):
    border.forward(GRID_SIZE * CELL_SIZE)
    border.right(90)
border.penup()

# ── Game logic (headless environment) ─────────────────────
env = SnakeEnv()
env.reset()

# ── Rendering objects ─────────────────────────────────────
score = Scoreboard()

snake_turtles = []


SHAPE_SCALE = CELL_SIZE / 20


def _get_segment_turtle(index):
    while index >= len(snake_turtles):
        t = Turtle(shape="square")
        t.color("white")
        t.penup()
        t.shapesize(stretch_len=SHAPE_SCALE, stretch_wid=SHAPE_SCALE)
        snake_turtles.append(t)
    return snake_turtles[index]


food_turtle = Turtle(shape="circle")
food_turtle.penup()
food_turtle.shapesize(stretch_len=SHAPE_SCALE * 0.5, stretch_wid=SHAPE_SCALE * 0.5)
food_turtle.color("yellow")

# ── Input handling ────────────────────────────────────────
pending_action = RIGHT


def press_up():
    global pending_action
    pending_action = UP


def press_down():
    global pending_action
    pending_action = DOWN


def press_left():
    global pending_action
    pending_action = LEFT


def press_right():
    global pending_action
    pending_action = RIGHT


screen.listen()
screen.onkey(press_up, "Up")
screen.onkey(press_down, "Down")
screen.onkey(press_left, "Left")
screen.onkey(press_right, "Right")


# ── Render helper ─────────────────────────────────────────
def render():
    for i, (r, c) in enumerate(env.snake):
        t = _get_segment_turtle(i)
        t.goto(*grid_to_pixel(r, c))
        t.showturtle()
    for i in range(len(env.snake), len(snake_turtles)):
        snake_turtles[i].hideturtle()

    if env.food is not None:
        food_turtle.goto(*grid_to_pixel(*env.food))
        food_turtle.showturtle()

    score.score = env.score
    score.update_scoreboard()
    screen.update()


# ── Main game loop ────────────────────────────────────────
try:
    while True:
        _, _, done = env.step(pending_action)
        render()
        time.sleep(get_step_delay(env.score))

        if done:
            score.reset_scoreboard()
            time.sleep(1)
            env.reset()
            pending_action = RIGHT
except (tkinter.TclError, turtle.Terminator):
    pass
