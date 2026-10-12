#Part 0h to 1h. A very basic RPS game

import random

def get_move():
    player_move=str(input("MAKE A MOVE (ROCK,PAPER OR SCISSORS): ")).strip().upper()
    options=[
        'ROCK',
        'SCISSORS',
        'PAPER'
             ]
    computer_move=random.choice(options)
    choices={
        "player":player_move,
        "computer":computer_move
        }
    return choices

def check_move(player_move,computer_move):
    
    if player_move==computer_move:
        return f"tie since computer: {computer_move} matches against your move: {player_move}"

    elif player_move == "ROCK":
        return "You won!" if computer_move=="SCISSORS" else "You lost!"

    elif player_move=="PAPER":
        return "You won!" if computer_move=="ROCK" else "You lost!"

    elif player_move=="SCISSORS":
        return "You won!" if computer_move=="PAPER" else "You lost!"

    else:
        return "ENTER A PROPER CHOICE"

moves=get_move()
outcome=check_move(moves["player"], moves["computer"])

