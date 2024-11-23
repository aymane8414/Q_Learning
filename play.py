import pygame
import numpy as np
import pickle
import os

pygame.init()

WIDTH, HEIGHT = 800, 800
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q Learning Car Simulation - Play Mode")

# Paramètres
MAX_SENSOR_DISTANCE = 250
STATE = [10, 10, 10]  # Nombre de divisions pour les capteurs
ACTIONS = [0, 1, 2]  # gauche, tout droit, droite
Q_TABLE_FILE = "q_table.pkl"

# Fonctions utilitaires
def draw_track():
    # Nouvelle map avec le chemin en spirale
    pygame.draw.line(screen, GREEN, (100, 100), (700, 100), 10)
    pygame.draw.line(screen, GREEN, (700, 100), (700, 700), 10)
    pygame.draw.line(screen, GREEN, (700, 700), (100, 700), 10)
    pygame.draw.line(screen, GREEN, (100, 700), (100, 200), 10)
    pygame.draw.line(screen, GREEN, (100, 200), (600, 200), 10)
    pygame.draw.line(screen, GREEN, (600, 200), (600, 600), 10)
    pygame.draw.line(screen, GREEN, (600, 600), (200, 600), 10)
    pygame.draw.line(screen, GREEN, (200, 600), (200, 300), 10)
    pygame.draw.line(screen, GREEN, (200, 300), (500, 300), 10)
    pygame.draw.line(screen, GREEN, (500, 300), (500, 500), 10)

def draw_text(surface, text, size, color, x, y):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(topleft=(x, y))
    surface.blit(text_surface, text_rect)

class Car:
    def __init__(self):
        self.x, self.y = WIDTH // 2, HEIGHT // 2
        self.angle = 0
        self.speed = 3
        self.sensors = [0, 0, 0]

    def move(self, action):
        if action == 0:
            self.angle -= 5
        elif action == 2:
            self.angle += 5

        radian_angle = np.radians(self.angle)
        self.x += int(self.speed * np.cos(radian_angle))
        self.y -= int(self.speed * np.sin(radian_angle))

    def draw(self):
        car_rect = pygame.Rect(self.x - 15, self.y - 15, 30, 30)
        pygame.draw.rect(screen, RED, car_rect)

    def sense(self):
        radian_angle = np.radians(self.angle)
        directions = [radian_angle - np.pi / 4, radian_angle, radian_angle + np.pi / 4]
        max_distance = MAX_SENSOR_DISTANCE

        for i, direction in enumerate(directions):
            for d in range(max_distance):
                x = self.x + int(d * np.cos(direction))
                y = self.y - int(d * np.sin(direction))
                if x < 100 or x > WIDTH - 100 or y < 100 or y > HEIGHT - 100:
                    self.sensors[i] = d / max_distance
                    break
            else:
                self.sensors[i] = 1
        return self.sensors

    def reset(self):
        self.x, self.y = WIDTH // 2, HEIGHT // 2
        self.angle = 0
        self.sensors = [0, 0, 0]

class QLearningPlayer:
    def __init__(self):
        self.q_table = self.load_q_table()

    def discretize_state(self, state):
        bins = [np.linspace(0, 1, num=b) for b in STATE]
        discrete_state = tuple(np.digitize(s, bins[i]) - 1 for i, s in enumerate(state))
        return discrete_state

    def choose_action(self, state):
        discrete_state = self.discretize_state(state)
        return np.argmax(self.q_table[discrete_state])

    def load_q_table(self):
        if os.path.exists(Q_TABLE_FILE):
            with open(Q_TABLE_FILE, "rb") as f:
                print("Q-table chargée depuis le fichier.")
                return pickle.load(f)
        raise FileNotFoundError("Q-table introuvable. Assurez-vous que la table a été entraînée et sauvegardée.")

def play():
    clock = pygame.time.Clock()
    car = Car()
    player = QLearningPlayer()

    episode = 1
    score = 0
    best_score = 0

    running = True
    while running:
        screen.fill(WHITE)
        draw_track()
        car.draw()

        state = car.sense()
        action = player.choose_action(state)
        car.move(action)

        if car.x < 100 or car.x > WIDTH - 100 or car.y < 100 or car.y > HEIGHT - 100:
            reward = -10
            done = True
        else:
            reward = 1
            done = False

        score += reward

        if done:
            if score > best_score:
                best_score = score
            car.reset()
            print(f"Épisode {episode} terminé avec un score de {score}. Meilleur score: {best_score}")
            episode += 1
            score = 0

        draw_text(screen, f"Épisode: {episode}", 24, BLACK, 10, HEIGHT - 100)
        draw_text(screen, f"Score: {score}", 24, BLACK, 10, HEIGHT - 80)
        draw_text(screen, f"Meilleur score: {best_score}", 24, BLACK, 10, HEIGHT - 60)

        pygame.display.flip()
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

if __name__ == "__main__":
    play()
