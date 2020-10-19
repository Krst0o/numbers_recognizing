import pygame

# Define colors
black = (0, 0, 0)
darkgray = (50, 50, 50)
white = (255, 255, 255)
lightgray = (170, 170, 170)
# Set block seizures
block_width = 50
block_height = 50
margin = 1
window_size = (256, 400)

# Create buttons grid
grid = [5*[0] for _ in range(5)]

# Initialization pygame
pygame.init()
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Numbers - neural network")


# Draw grid
def draw_grid():
    for row in range(5):
        for col in range(5):
            color = white
            if grid[row][col] == 1:
                color = darkgray
            pygame.draw.rect(screen, color, [(block_width) * col,
                                             (block_height) * row,
                                             block_width,
                                             block_height])
    clear_button = pygame.draw.rect(screen, lightgray, (1, 270, 50, 25))
    inverse_button = pygame.draw.rect(screen, lightgray, (52, 270, 50, 25))

def negative_values():
    for row in range(5):
        for col in range(5):
            if grid[row][col] == 1:
                grid[row][col] = 0
            else:
                grid[row][col] = 1


def change_clicked_button(clicked_x, clicked_y):
    # Get row and column
    row = int(clicked_y / 50)
    col = int(clicked_x / 50)
    # Change color and value
    if (row >= 0 and row <= 4) and (col >= 0  and col <= 4):
        if grid[row][col] == 0:
            grid[row][col] = 1
        else:
            grid[row][col] = 0

# Main loop
while True:
    # Update grid
    pygame.display.update()
    draw_grid()
    # Listen for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                clicked_x = pygame.mouse.get_pos()[0]
                clicked_y = pygame.mouse.get_pos()[1]
                change_clicked_button(clicked_x, clicked_y)
        else:
            continue
