import pygame
import numpy as np
import random
# import training inputs and variables
import data
from variables import *

# GUI INIT
pygame.init()
font = pygame.font.SysFont('comicsansms', 20)
label_font = pygame.font.SysFont('comicsansms', 28)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Numbers - neural network")
perceptrons = []
which_perceptron = ""
grid = np.array(25*[0])

# PERCEPTRON CLASS
class Perceptron(object):
    def __init__(self, no_of_inputs, perceptron_value, learning_rate=0.01, iterations=10000):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.no_of_inputs = no_of_inputs
        self.perceptron_value = perceptron_value
        self.weights = np.random.rand(self.no_of_inputs + 1) - 0.5
        self.best_weights = np.copy(self.weights)

    def train(self, training_data, labels):
        lifetime = 0
        record = 0
        for _ in range(self.iterations):
            is_proper_example = 0
            # ZADANIE DOMOWE - losowosc
            random_input = random.choice(list(zip(training_data, labels)))
            random_training_data = np.copy(random_input[0])
            random_training_label = np.copy(random_input[1])

            # ZADANIE DOMOWE - zaburzenie wejscia
            noise = np.random.binomial(1, 0.01, size=(25))
            random_training_data += noise
            random_training_data = np.where(random_training_data > 0, 1, 0)

            if random_training_label == self.perceptron_value:
                is_proper_example = 1

            prediction = self.output(random_training_data)
            if (is_proper_example - prediction) == 0:
                lifetime += 1
                if (lifetime > record):
                    self.best_weights = np.copy(self.weights)
            else:
                self.weights[1:] += self.learning_rate * (is_proper_example - prediction) * random_training_data
                self.weights[0] += self.learning_rate * (is_proper_example - prediction)
                self.lifetime = 0

                # ZADANIE DOMOWE 3 - warunek stopu
                # ZADANIE DOMOWE 4 - PLA + RPLA
    
    def output(self, input):
        summation = np.dot(self.best_weights[1:], input) + self.best_weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

# Change button color and gird value after click on proper button
def change_clicked_button(clicked_x, clicked_y):
    global grid
    # Get row and column
    row = int(clicked_y / 51)
    col = int(clicked_x / 51)
    i = 5 * row + col
    # Change color and value
    if grid[i] == 0:
        grid[i] = 1
    else:
        grid[i] = 0

# Draw grid
def draw_grid():
    global grid
    for i in range(25):
        color = white
        if grid[i] == 1:
            color = darkgray
        pygame.draw.rect(screen, color, [(block_width + 1) * int(i%5), (block_height + 1) * int(i/5),
                                        block_width, block_height])
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
    # Create blur button with text
    blur_button_text = font.render("BLUR", False, (0, 0, 0))
    blur_button = pygame.draw.rect(screen, lightgray, (195, 270, 64, 32))
    screen.blit(blur_button_text, blur_button)

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
    # Create exit button with text
    exit_button_text = font.render("EXIT", False, (0, 0, 0))
    exit_button = pygame.draw.rect(screen, lightgray, (195, 303, 64, 32))
    screen.blit(exit_button_text, exit_button)

    ### THIRD ROW
    # Create learn button
    learn_button_text = font.render("LEARN", False, (0, 0, 0))
    learn_button = pygame.draw.rect(screen, lightgray, (0, 336, 64, 32))
    screen.blit(learn_button_text, learn_button)
    # Create check button
    learn_button_text = font.render("CHECK", False, (0, 0, 0))
    learn_button = pygame.draw.rect(screen, lightgray, (65, 336, 64, 32))
    screen.blit(learn_button_text, learn_button)

    # FOURTH ROW
    # Create label
    label = label_font.render(which_perceptron, 0, (255, 255, 255))
    screen.blit(label, (0, 380))

# Set grid values to 0
def clear_grid_button():
    global grid
    global which_perceptron
    for i in range(25):
        grid[i] = 0

# Turn over grid values
def inverse_grid_button():
    global grid
    for i in range(25):
        if grid[i] == 0:
            grid[i] = 1
        else:
            grid[i] = 0

# Change random button value
def blur_grid_button():
    global grid
    noise = np.random.binomial(1, 0.1, size=(5,5))
    grid += noise.reshape((-1))
    grid = np.where(grid > 0, 1, 0)

# Shift array
def up_grid_button():
    global grid
    grid = grid.reshape((5, 5))
    grid = np.roll(grid, -1, axis=0)
    grid = grid.reshape((-1))

# Shift array
def down_grid_button():
    global grid
    grid = grid.reshape((5,5))
    grid = np.roll(grid, 1, axis=0)
    grid = grid.reshape((-1))

# Shift array
def left_grid_button():
    global grid
    grid = grid.reshape((5, 5))
    grid = np.roll(grid, -1, axis=1)
    grid = grid.reshape((-1))

# Shift array
def right_grid_button():
    global grid
    grid = grid.reshape((5, 5))
    grid = np.roll(grid, 1, axis=1)
    grid = grid.reshape((-1))

# Read data from file and train perceptrons
def learn_grid_button():
    training_inputs = [ np.ravel(n) for n in data.number ]
    training_data = [np.ravel(n) for n in np.array(training_inputs)[:, 1]]
    labels = [np.ravel(n) for n in np.array(training_inputs)[:, 0]]
    for i in range(10):
        perceptrons[i].train(training_data, labels)
    print("Learned")


# Check number classification for selected values in grid
def check_grid_button():
    global which_perceptron
    screen.fill((0, 0, 0))
    which_perceptron = "Perceptron: "
    for i in range(10):
        print("Perceptron {}: {}".format(i, perceptrons[i].output(grid)))
        if(perceptrons[i].output(grid) == 1):
            which_perceptron += str(i)
# Main loop
def main():
    for i in range(10):
        perceptrons.append(Perceptron(5 * 5, i))
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
                    # Clicked blur button
                    if (clicked_x >= 195 and clicked_x <= 259) and (clicked_y >= 270 and clicked_y < 302):
                        blur_grid_button()
                    # Clicked left button
                    if (clicked_x >= 0 and clicked_x <= 64) and (clicked_y >= 303 and clicked_y <= 335):
                        left_grid_button()
                    # Clicked down button
                    if (clicked_x >= 65 and clicked_x <= 129) and (clicked_y >= 303 and clicked_y <= 335):
                        down_grid_button()
                    # Clicked right button
                    if (clicked_x >= 130 and clicked_x <= 194) and (clicked_y >= 303 and clicked_y <= 335):
                        right_grid_button()
                    # Clicked exit button
                    if (clicked_x >= 195 and clicked_x <= 259) and (clicked_y >= 303 and clicked_y <= 335):
                        pygame.quit()
                    # Clicked learn button
                    if (clicked_x >= 0 and clicked_x <= 64) and (clicked_y >= 336 and clicked_y <= 368):
                        learn_grid_button()
                    # Clicked check button
                    if(clicked_x >= 65 and clicked_x <= 129) and (clicked_y >= 336 and clicked_y <= 368):
                        check_grid_button()
            else:
                continue
main()
