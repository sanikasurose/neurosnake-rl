import time
import tkinter
import turtle
from turtle import Screen, Turtle

from snake_env import SnakeEnv, UP, DOWN, LEFT, RIGHT
from score import Scoreboard

CELL_SIZE = 20
GRID_CENTER = 7


def grid_to_pixel(row, col):
    x = (col - GRID_CENTER) * CELL_SIZE
    y = (GRID_CENTER - row) * CELL_SIZE
    return x, y


# ── Screen setup ──────────────────────────────────────────
screen = Screen()
screen.setup(width=600, height=600)
screen.bgcolor("black")
screen.title("Funky Snake Game")
screen.tracer(0)

# ── Game logic (headless environment) ─────────────────────
env = SnakeEnv()
env.reset()

# ── Rendering objects ─────────────────────────────────────
score = Scoreboard()

snake_turtles = []


def _get_segment_turtle(index):
    while index >= len(snake_turtles):
        t = Turtle(shape="square")
        t.color("white")
        t.penup()
        snake_turtles.append(t)
    return snake_turtles[index]


food_turtle = Turtle(shape="circle")
food_turtle.penup()
food_turtle.shapesize(stretch_len=0.5, stretch_wid=0.5)
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
        time.sleep(0.05)

        if done:
            score.reset_scoreboard()
            env.reset()
            pending_action = RIGHT
except (tkinter.TclError, turtle.Terminator):
    pass
