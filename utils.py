import pygame
import os
import random 
import numpy as np
from q_learning import *

WIDTH, HEIGHT = 800, 800
BLACK = (0, 0, 0)


def generate_random_obstacles(num_obstacles, radius=250, min_spacing=50, min_start_distance=100):
    obstacles = []
    start_x, start_y = WIDTH // 2, HEIGHT // 2 - radius

    while len(obstacles) < num_obstacles:
        angle = random.uniform(0, 2 * np.pi)
        x = int(WIDTH // 2 + radius * np.cos(angle))
        y = int(HEIGHT // 2 + radius * np.sin(angle))

        if all(np.hypot(x - other.centerx, y - other.centery) >= min_spacing for other in obstacles) and \
           np.hypot(x - start_x, y - start_y) >= min_start_distance:
            obstacles.append(pygame.Rect(x - 10, y - 10, 20, 20))
    return obstacles


def draw_obstacles(screen, obstacles):
    for obstacle in obstacles:
        pygame.draw.rect(screen, BLACK, obstacle)


def draw_reward(screen,reward, color, position=(50, 50)):
    pygame.draw.circle(screen, color, position, 10)
    screen.blit(pygame.font.Font(None, 24).render(f"{reward}", True, (0, 0, 0)), (position[0] + 15, position[1] - 5))


def load_scores():
    if os.path.exists("scores.txt"):
        with open("scores.txt", "r") as f:
            best_score = int(f.readline().strip())
            episode = int(f.readline().strip())
        return best_score, episode
    return 0, 0


def save_scores(best_score, episode):
    with open("scores.txt", "w") as f:
        f.write(f"{best_score}\n{episode}\n")


def draw_text(surface, text, size, color, x, y):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, (x, y))
    
# def draw_track():
#     # case finale (zone jaune)
#     pygame.draw.line(screen, BLACK, (WIDTH // 2 - 5, HEIGHT // 2 - 300), (WIDTH // 2 - 5, HEIGHT // 2 - 190), 10)  # ligne rouge
    
#     # circuit
#     pygame.draw.line(screen, GREEN, (WIDTH // 2 + 5, HEIGHT // 2 - 300), (WIDTH // 2 + 5, HEIGHT // 2 - 190), 10)  # ligne noir
#     pygame.draw.circle(screen, BLACK, (WIDTH // 2, HEIGHT // 2), 300, 10)  # extérieur
#     pygame.draw.circle(screen, BLACK, (WIDTH // 2, HEIGHT // 2), 200, 10)  # intérieure
