import pygame
import numpy as np
from main import *

WIDTH, HEIGHT = 800, 800
RED = (255, 0, 0)
MAX_SENSOR_DISTANCE = 100
MAX_SENSOR_DISTANCE_CENTER = 50

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class Car:
    def __init__(self):
        self.x, self.y = WIDTH // 2 - 25, HEIGHT // 2 - 250
        self.angle = 200
        self.speed = 2
        self.sensors = [0, 0, 0]

    def move(self, action):
        if action == 0:
            self.angle -= 2.5
        elif action == 2:
            self.angle += 2.5

        radian_angle = np.radians(self.angle)
        self.x += int(self.speed * np.cos(radian_angle))
        self.y -= int(self.speed * np.sin(radian_angle))

    def draw(self):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), 10)

    def draw_sensors(self):
        radian_angle = np.radians(self.angle)
        directions = [
            radian_angle - np.pi / 3,  # Capteur gauche
            radian_angle,              # Capteur central
            radian_angle + np.pi / 3   # Capteur droit
        ]

        for i, direction in enumerate(directions):
            if i == 1:  # Capteur central (binaire)
                end_x = self.x + int(MAX_SENSOR_DISTANCE_CENTER * np.cos(direction))
                end_y = self.y - int(MAX_SENSOR_DISTANCE_CENTER * np.sin(direction))
                pygame.draw.line(screen, GREEN if self.sensors[i] == 1 else RED, (self.x, self.y), (end_x, end_y), 2)

                # Indiquer la détection (ON/OFF)
                color = GREEN if self.sensors[i] == 1 else RED
                state_text = "ON" if self.sensors[i] == 1 else "OFF"

                # Dessiner un cercle indiquant l'état
                y_offset = 50 + i * 60
                pygame.draw.circle(screen, color, (WIDTH - 50, y_offset), 10)

                # Afficher le texte (ON/OFF)
                font = pygame.font.Font(None, 16)
                text_surface = font.render(state_text, True, BLACK)
                screen.blit(text_surface, (WIDTH - 80, y_offset - 10))
            else:  # Capteurs gauche et droit (mesure de distance)
                end_x = self.x + int(self.sensors[i] * MAX_SENSOR_DISTANCE * np.cos(direction))
                end_y = self.y - int(self.sensors[i] * MAX_SENSOR_DISTANCE * np.sin(direction))

                # Dessiner la ligne du capteur
                pygame.draw.line(screen, BLUE, (self.x, self.y), (end_x, end_y), 2)

                # Dessiner un cercle pour indiquer la distance
                y_offset = 50 + i * 60
                color = GREEN if self.sensors[i] == 1 else RED
                pygame.draw.circle(screen, color, (WIDTH - 50, y_offset), 10)

                # Afficher la distance ou "-" si au maximum
                distance_text = f"{self.sensors[i] * MAX_SENSOR_DISTANCE:.1f}" if self.sensors[i] < 1 else "-"
                font = pygame.font.Font(None, 16)
                text_surface = font.render(distance_text, True, BLACK)
                screen.blit(text_surface, (WIDTH - 80, y_offset - 10))


    def sense(self,obstacles):
        radian_angle = np.radians(self.angle)
        directions = [
            radian_angle - np.pi / 3,  # Capteur gauche
            radian_angle,              # Capteur central
            radian_angle + np.pi / 3   # Capteur droit
        ]

        for i, direction in enumerate(directions):
            max_distance = MAX_SENSOR_DISTANCE if i != 1 else MAX_SENSOR_DISTANCE_CENTER

            if i == 1:  # Capteur central (binaire)
                for d in range(max_distance):
                    x = self.x + int(d * np.cos(direction))
                    y = self.y - int(d * np.sin(direction))

                    if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
                        self.sensors[i] = 0  # Pas d'obstacle
                        break

                    if (
                        (WIDTH // 2 - 300 <= x <= WIDTH // 2 + 300 and HEIGHT // 2 - 300 <= y <= HEIGHT // 2 + 300)
                        and ((x - WIDTH // 2) ** 2 + (y - HEIGHT // 2) ** 2 >= 300 ** 2)
                    ) or (
                        (WIDTH // 2 - 200 <= x <= WIDTH // 2 + 200 and HEIGHT // 2 - 200 <= y <= HEIGHT // 2 + 200)
                        and ((x - WIDTH // 2) ** 2 + (y - HEIGHT // 2) ** 2 <= 200 ** 2)
                    ) or any(obstacle.collidepoint(x, y) for obstacle in obstacles):  # Vérification des obstacles
                        self.sensors[i] = 1  # Obstacle détecté
                        break

                else:
                    self.sensors[i] = 0  # Pas d'obstacle
            else:  # Capteurs gauche et droit (distance)
                for d in range(max_distance):
                    x = self.x + int(d * np.cos(direction))
                    y = self.y - int(d * np.sin(direction))

                    if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
                        self.sensors[i] = d / max_distance
                        break

                    if (
                        (WIDTH // 2 - 300 <= x <= WIDTH // 2 + 300 and HEIGHT // 2 - 300 <= y <= HEIGHT // 2 + 300)
                        and ((x - WIDTH // 2) ** 2 + (y - HEIGHT // 2) ** 2 >= 300 ** 2)
                    ) or (
                        (WIDTH // 2 - 200 <= x <= WIDTH // 2 + 200 and HEIGHT // 2 - 200 <= y <= HEIGHT // 2 + 200)
                        and ((x - WIDTH // 2) ** 2 + (y - HEIGHT // 2) ** 2 <= 200 ** 2)
                    ):
                        self.sensors[i] = d / max_distance
                        break
                else:
                    self.sensors[i] = 1

        return self.sensors

    def reset(self):
        self.x, self.y = WIDTH // 2 - 25, HEIGHT // 2 - 250
        self.angle = 200
        self.speed = 2
        self.sensors = [0, 0, 0]
