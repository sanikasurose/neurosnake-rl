from turtle import Screen
from snake import Snake
from score import Scoreboard
import time
from food import Food

screen = Screen()
screen.setup(width=600, height=600)
screen.bgcolor("black")
screen.title("Funky Snake Game")
screen.tracer(0)

snake = Snake()
food = Food()
score = Scoreboard()

screen.listen()
screen.onkey(snake.up, "Up")
screen.onkey(snake.down, "Down")
screen.onkey(snake.left, "Left")
screen.onkey(snake.right, "Right")

game_on = True
while game_on:
    screen.update()
    time.sleep(0.05)

    snake.move()

    #Detect collision with food
    if snake.head.distance(food) < 15:
        food.refresh()
        snake.extend()
        score.increase_score()

    #Detect collision with wall
    if (snake.head.xcor() > 290 or snake.head.xcor() < -290 or
            snake.head.ycor() > 290 or snake.head.ycor() < -290):
        score.reset_scoreboard()
        snake.reset()

    #Detect collision with tail
    for segment in snake.segments[1:]:
        if segment == snake.head:
            pass
        elif snake.head.distance(segment) < 10:
            score.reset_scoreboard()
            snake.reset()

screen.exitonclick()
