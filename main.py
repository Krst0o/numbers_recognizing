import pygame
import numpy as np
import random

# TRAINING DATA
number = [[] for _ in range(10)]
number[0] = [
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0]
  ]
number[1] = [
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0]
  ]
number[2] = [
    [0.0, 1.0, 1.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0]
  ]
number[3] = [
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0]
  ]
number[4] = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0]
  ]
number[5] = [
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0]
  ]
number[6] = [
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0]
  ]
number[7] = [
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0]
  ]
number[8] = [
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0]
  ]
number[9] = [
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0]
  ]
training_inputs = [ np.ravel(n) for n in number ]
perceptrons = []

# PERCEPTRON CLASS
class Perceptron(object):
    def __init__(self, no_of_inputs, learning_rate=0.01, iterations=1000):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.no_of_inputs = no_of_inputs
        self.weights = np.random.rand(self.no_of_inputs + 1) - 0.5
        self.best_weights = self.weights
    
    def train(self, training_data, labels):
        lifetime = 0
        record = 0
        for _ in range(self.iterations):
            # ZADANIE DOMOWE - losowosc
            random_input = random.choice(list(zip(training_data, labels)))
            # input = noisy(input) # ZADANIE DOMOWE - zaburzenie wejscia
            """
            Zamienic inputy na numpy array i na nich wykonywać operacje 
      
            random_input[0] += np.random.binomial(1, 0.1, size=(25))
            random_input[0] = np.where(random_input[0] > 0, 1, 0)
            """
            prediction = self.output(random_input[0])
            if (random_input[1] - prediction) == 0:
                lifetime += 1
                if (lifetime > record):
                    self.best_weights = self.weights
            else:
                self.weights[1:] += self.learning_rate * (random_input[1] - prediction) * random_input[0]
                self.weights[0] += self.learning_rate * (random_input[1] - prediction)
                lifetime = 0
    
            # ZADANIE DOMOWE 3 - warunek stopu
            # ZADANIE DOMOWE 4 - PLA + RPLA
            # h(x) = (ax + b) % m
    
    def output(self, input):
        summation = np.dot(self.best_weights[1:], input) + self.best_weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

# GUI
# Define colors
pygame.init()
black = (0, 0, 0)
darkgray = (50, 50, 50)
gray = (140, 140, 140)
white = (255, 255, 255)
lightgray = (170, 170, 170)
# Set block seizures
block_width = 50
block_height = 50
margin = 1
window_size = (256, 400)
font = pygame.font.SysFont('comicsansms', 22)

# Create buttons grid
grid = [5*[0] for _ in range(5)]

# Initialization pygame
pygame.init()
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Numbers - neural network")

def negative_values():
    for row in range(5):
        for col in range(5):
            if grid[row][col] == 1:
                grid[row][col] = 0
            else:
                grid[row][col] = 1


def change_clicked_button(clicked_x, clicked_y):
    # Get row and column
    row = int(clicked_y / 51)
    col = int(clicked_x / 51)
    # Change color and value
    if grid[row][col] == 0:
        grid[row][col] = 1
    else:
        grid[row][col] = 0

# Draw grid
def draw_grid():
    for row in range(5):
        for col in range(5):
            color = white
            if grid[row][col] == 1:
                color = darkgray
            pygame.draw.rect(screen, color, [(block_width + 1) * col,
                                             (block_height + 1) * row,
                                             block_width,
                                             block_height])
    ### FIRST ROW
    # Create clear button with text
    clear_button_text = font.render("CLEAR", False, (0, 0, 0))
    clear_button = pygame.draw.rect(screen, lightgray, (0, 270, 64, 32))
    screen.blit(clear_button_text, clear_button)
    # Create up button with text
    up_button_text = font.render("UP", False, (0, 0, 0))
    up_button = pygame.draw.rect(screen, gray, (65, 270, 64, 32))
    screen.blit(up_button_text, up_button)
    # Create inverse button with text
    inverse_button_text = font.render("INVERSE", False, (0, 0, 0))
    inverse_button = pygame.draw.rect(screen, lightgray, (130, 270, 64, 32))
    screen.blit(inverse_button_text, inverse_button)
    ### SECOND ROW
    # Create left button with text
    left_button_text = font.render("LEFT", False, (0, 0, 0))
    left_button = pygame.draw.rect(screen, gray, (0, 303, 64, 32))
    screen.blit(left_button_text, left_button)
    # Create down button with text
    down_button_text = font.render("DOWN", False, (0, 0, 0))
    down_button = pygame.draw.rect(screen, gray, (65, 303, 64, 32))
    screen.blit(down_button_text, down_button)
    # Create right button with text
    right_button_text = font.render("RIGHT", False, (0, 0, 0))
    right_button = pygame.draw.rect(screen, gray, (130, 303, 64, 32))
    screen.blit(right_button_text, right_button)
    ### THIRD ROW
    # Create learn button
    learn_button_text = font.render("LEARN", False, (0, 0, 0))
    learn_button = pygame.draw.rect(screen, gray, (0, 336, 64, 32))
    screen.blit(learn_button_text, learn_button)
    # Create check button
    learn_button_text = font.render("CHECK", False, (0, 0, 0))
    learn_button = pygame.draw.rect(screen, gray, (65, 336, 64, 32))
    screen.blit(learn_button_text, learn_button)
def clear_grid_button():
    for row in range(5):
        for col in range(5):
            grid[row][col] = 0

def inverse_grid_button():
    for row in range(5):
        for col in range(5):
            if grid[row][col] == 0:
                grid[row][col] = 1
            else:
                grid[row][col] = 0
def up_grid_button():
    print("Up button")

def down_grid_button():
    print("down button")

def left_grid_button():
    print("left button")

def right_grid_button():
    print("right button")

def learn_grid_button():
    for i in range(10):
        labels = np.zeros(10)
        labels[i] = 1
        perceptrons[i].train(training_inputs, labels)
        print(perceptrons[i].best_weights)

def check_grid_button():
    for i in range(10):
        print(perceptrons[i].output(grid))


# Main loop
def main():
    for _ in range(10):
        perceptrons.append(Perceptron(5 * 5))
    while True:
        # Update grid
        pygame.display.update()
        draw_grid()
        # Listen for events
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    clicked_x = pygame.mouse.get_pos()[0]
                    clicked_y = pygame.mouse.get_pos()[1]
                    # Clicked grid
                    if (clicked_x > 0 and clicked_x < 256) and  (clicked_y > 0 and clicked_y < 256):
                        change_clicked_button(clicked_x, clicked_y)
                    # Clicked clear button
                    if (clicked_x >= 0 and clicked_x <= 64) and (clicked_y >= 270 and clicked_y <= 302):
                        clear_grid_button()
                    # Clicked up button
                    if (clicked_x >= 65 and clicked_x <= 129) and (clicked_y >= 270 and clicked_y < 302):
                        up_grid_button()
                    # Clicked inverse button
                    if (clicked_x >= 130 and clicked_x <= 194) and (clicked_y >= 270 and clicked_y < 302):
                        inverse_grid_button()
                    # Clicked left button
                    if (clicked_x >= 0 and clicked_x <= 64) and (clicked_y >= 303 and clicked_y <= 335):
                        left_grid_button()
                    # Clicked down button
                    if (clicked_x >= 65 and clicked_x <= 129) and (clicked_y >= 303 and clicked_y <= 335):
                        down_grid_button()
                    # Clicked right button
                    if (clicked_x >= 130 and clicked_x <= 194) and (clicked_y >= 303 and clicked_y <= 335):
                        right_grid_button()
                    if (clicked_x >= 0 and clicked_x <= 64) and (clicked_y >= 336 and clicked_y <= 368):
                        learn_grid_button()
                    if(clicked_x >= 65 and clicked_x <= 129) and (clicked_y >= 336 and clicked_y <= 368):
                        check_grid_button()
            else:
                continue
main()
