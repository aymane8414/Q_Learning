import pygame
import numpy as np
import random
import os
import pickle

pygame.init()

WIDTH, HEIGHT = 800, 800
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q Learning car simulation")

# parametres
ALPHA = 0.1  # taux d'apprentissage
GAMMA = 0.99  # facteur de récompense future
EPSILON = 0.99  # exploration initiale
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01

# discrétisation des états
STATE = [10, 10, 10]  # nombre de divisions pour les capteurs
MAX_SENSOR_DISTANCE = 250

# Actions
ACTIONS = [0, 1, 2]  # gauche, tout droit, droite

Q_TABLE_FILE = "q_table.pkl"  # Nom du fichier pour sauvegarder la table Q

# Fonctions utilitaires
def draw_track():
    # Nouvelle map avec le chemin en spirale
    pygame.draw.line(screen, GREEN, (100, 100), (700, 100), 10)
    pygame.draw.line(screen, GREEN, (700, 100), (700, 700), 10)
    pygame.draw.line(screen, GREEN, (700, 700), (100, 700), 10)
    pygame.draw.line(screen, GREEN, (100, 700), (100, 200), 10)
    pygame.draw.line(screen, GREEN, (400, 600), (400, 300), 10) #test
    pygame.draw.line(screen, GREEN, (100, 200), (600, 200), 10)
    pygame.draw.line(screen, GREEN, (600, 200), (600, 600), 10)
    pygame.draw.line(screen, GREEN, (600, 600), (200, 600), 10)
    pygame.draw.line(screen, GREEN, (200, 600), (200, 300), 10)
    pygame.draw.line(screen, GREEN, (200, 300), (500, 300), 10)
    pygame.draw.line(screen, GREEN, (500, 300), (500, 500), 10)

    # case final
    pygame.draw.rect(screen, (255, 255, 0), (100, 100, 50, 100))

def draw_text(surface, text, size, color, x, y):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(topleft=(x, y))
    surface.blit(text_surface, text_rect)

class Car:
    def __init__(self):
        self.x, self.y = WIDTH // 2, HEIGHT // 2
        self.angle = 0
        self.speed = 2
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

    def draw_sensors(self):
        radian_angle = np.radians(self.angle)
        directions = [radian_angle - np.pi / 4, radian_angle, radian_angle + np.pi / 4]

        for i, direction in enumerate(directions):
            end_x = self.x + int(self.sensors[i] * MAX_SENSOR_DISTANCE * np.cos(direction))
            end_y = self.y - int(self.sensors[i] * MAX_SENSOR_DISTANCE * np.sin(direction))
            pygame.draw.line(screen, BLUE, (self.x, self.y), (end_x, end_y), 2)

            y_offset = 50 + i * 60
            if self.sensors[i] < 1:
                color = RED 
                distance_text = f"{self.sensors[i] * MAX_SENSOR_DISTANCE:.1f}" 
            else:
                color = GREEN  
                distance_text = "-"  

            pygame.draw.circle(screen, color, (WIDTH - 50, y_offset), 10)

            # Afficher la distance à côté du cercle
            font = pygame.font.Font(None, 16)
            text_surface = font.render(distance_text, True, BLACK)
            screen.blit(text_surface, (WIDTH - 80, y_offset - 10))

    def sense(self):
        radian_angle = np.radians(self.angle)
        directions = [radian_angle - np.pi / 4, radian_angle, radian_angle + np.pi / 4]
        max_distance = MAX_SENSOR_DISTANCE

        for i, direction in enumerate(directions):
            for d in range(max_distance):
                x = self.x + int(d * np.cos(direction))
                y = self.y - int(d * np.sin(direction))

                if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
                    self.sensors[i] = d / max_distance
                    break

        
                if screen.get_at((x, y))[:3] == GREEN:  
                    self.sensors[i] = d / max_distance
                    break
            else:
                self.sensors[i] = 1
        return self.sensors


    def reset(self):
        self.x, self.y = WIDTH // 2, HEIGHT // 2
        self.angle = 0
        self.sensors = [0, 0, 0]

class QLearningAgent:
    def __init__(self):
        self.q_table = self.load_q_table()

    def discretize_state(self, state):
        bins = [np.linspace(0, 1, num=b) for b in STATE]
        discrete_state = tuple(np.digitize(s, bins[i]) - 1 for i, s in enumerate(state))
        return discrete_state

    def choose_action(self, state):
        if random.random() < EPSILON:
            return random.choice(ACTIONS)
        discrete_state = self.discretize_state(state)
        return np.argmax(self.q_table[discrete_state])

    def update_q_table(self, state, action, reward, next_state, done):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        q_value = self.q_table[discrete_state][action]
        max_next_q = np.max(self.q_table[discrete_next_state])

        target = reward + (GAMMA * max_next_q * (1 - done))
        self.q_table[discrete_state][action] += ALPHA * (target - q_value)

        self.save_q_table()

    def load_q_table(self):
        if os.path.exists(Q_TABLE_FILE):
            with open(Q_TABLE_FILE, "rb") as f:
                print("Q-table chargée depuis le fichier.")
                return pickle.load(f)
        print("Aucune Q-table trouvée. Initialisation d'une nouvelle table.")
        return np.zeros(tuple(STATE) + (len(ACTIONS),))

    def save_q_table(self):
        with open(Q_TABLE_FILE, "wb") as f:
            pickle.dump(self.q_table, f)

def main():
    global EPSILON
    clock = pygame.time.Clock()
    car = Car()
    agent = QLearningAgent()

    episode = 1
    score = 0
    best_score = 0

    running = True
    while running:
        screen.fill(WHITE)
        draw_track()
        car.draw()
        car.draw_sensors()

        state = car.sense()
        action = agent.choose_action(state)
        car.move(action)

        # detection de collision 
        collision = False
        car_rect = pygame.Rect(car.x - 15, car.y - 15, 30, 30)
        for x in range(car_rect.left, car_rect.right):
            for y in range(car_rect.top, car_rect.bottom):
                if 0 <= x < WIDTH and 0 <= y < HEIGHT: 
                    if screen.get_at((x, y))[:3] == GREEN:
                        collision = True
                        break
            if collision:
                break

        if (
            car.x < 0 or car.x >= WIDTH or car.y < 0 or car.y >= HEIGHT or collision
        ):
            reward = -10
            done = True
        else:
            reward = 1
            done = False

        yellow_zone = pygame.Rect(100, 100, 50, 100)
        if car_rect.colliderect(yellow_zone):
            reward = 100
            done = True
            print("case jaune atteinte")

        next_state = car.sense()
        agent.update_q_table(state, action, reward, next_state, done)

        score += reward

        if done:
            if score > best_score:
                best_score = score
            car.reset()
            print(f"\u00c9pisode {episode} termin\u00e9 avec un score de {score}. Meilleur score: {best_score}")
            episode += 1
            score = 0

            EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        draw_text(screen, f"\u00c9pisode: {episode}", 24, BLACK, 10, HEIGHT - 100)
        draw_text(screen, f"Score: {score}", 24, BLACK, 10, HEIGHT - 80)
        draw_text(screen, f"Meilleur score: {best_score}", 24, BLACK, 10, HEIGHT - 60)
        draw_text(screen, f"Epsilon: {EPSILON:.2f}", 24, BLACK, 10, HEIGHT - 40)

        pygame.display.flip()
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()


if __name__ == "__main__":
    main()
