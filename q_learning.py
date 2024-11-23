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
MAX_SENSOR_DISTANCE = 300

ACTIONS = [0, 1, 2]  # gauche, tout droit, droite

Q_TABLE_FILE = "q_table.pkl"  # fichier pour sauvegarder la table Q
SCORE_FILE = "scores.txt"

SCORE_FILE = "scores.txt"  # fichier pour stocker le meilleur score et les épisodes

def load_scores():
    if os.path.exists(SCORE_FILE):
        with open(SCORE_FILE, "r") as f:
            lines = f.readlines()
            best_score = int(lines[0].strip())
            episode = int(lines[1].strip())
            print(f"Scores chargés : Meilleur score = {best_score}, Épisode = {episode}")
            return best_score, episode
    print("Fichier scores.txt introuvable. Initialisation des scores à zéro.")
    return 0, 0

def save_scores(best_score, episode):
    with open(SCORE_FILE, "w") as f:
        f.write(f"{best_score}\n")
        f.write(f"{episode}\n")
    print(f"Scores sauvegardés : Meilleur score = {best_score}, Épisode = {episode}")


# fonctions utilitaires
def draw_track():
    # case finale (zone jaune)
    pygame.draw.line(screen, BLACK, (WIDTH // 2 - 5, HEIGHT // 2 - 300), (WIDTH // 2 - 5, HEIGHT // 2 - 190), 10)  # ligne rouge
    
    # circuit
    pygame.draw.line(screen, GREEN, (WIDTH // 2 + 5, HEIGHT // 2 - 300), (WIDTH // 2 + 5, HEIGHT // 2 - 190), 10)  # ligne verte
    pygame.draw.circle(screen, BLACK, (WIDTH // 2, HEIGHT // 2), 300, 10)  # extérieur
    pygame.draw.circle(screen, BLACK, (WIDTH // 2, HEIGHT // 2), 200)  # intérieure


def draw_text(surface, text, size, color, x, y):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(topleft=(x, y))
    surface.blit(text_surface, text_rect)

class Car:
    def __init__(self):
        self.x, self.y = WIDTH // 2 - 20, HEIGHT // 2 - 250  # position initiale de la voiture
        self.angle = 180
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

            # afficher la distance à côté du cercle
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
        self.x, self.y = WIDTH // 2 - 20, HEIGHT // 2 - 250

        self.angle = 180
        self.speed = 2
        self.sensors = [0, 0, 0]


class QLearningAgent:
    def __init__(self):
        self.q_table = self.load_q_table()

    def discretize_state(self, state):
        """
        Transforme les distances des capteurs en catégories
        """
        bins = [np.linspace(0, 1, num=b) for b in STATE]
        discrete_state = tuple(np.digitize(s, bins[i]) - 1 for i, s in enumerate(state))
        return discrete_state

    def choose_action(self, state):
        """
        Choisit une action pour la voiture :
        - Exploration : avec une probabilité < EPSILON, l'agent essaie une action au hasard
        - Exploitation : sinon, il utilise la table Q pour choisir l'action
        """
        if random.random() < EPSILON:
            return random.choice(ACTIONS)
        discrete_state = self.discretize_state(state)
        return np.argmax(self.q_table[discrete_state])

    def update_q_table(self, state, action, reward, next_state, done):
        """
        Met à jour la table Q en fonction de l'expérience actuelle
        - L'objectif est d'apprendre combien l'action effectuée était "bonne"
        """
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

    if not os.path.exists(Q_TABLE_FILE):
        print("Q-table introuvable. Réinitialisation des scores.")
        save_scores(0, 0)  # reinitialiser les scores à zéro
        best_score, episode = 0, 0
    else:
        # charger les scores et le nombre d'épisodes
        best_score, episode = load_scores()

    score = 0

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
        if (
            (car.x - WIDTH // 2) ** 2 + (car.y - HEIGHT // 2) ** 2 >= 300 ** 2
            or (car.x - WIDTH // 2) ** 2 + (car.y - HEIGHT // 2) ** 2 <= 200 ** 2
            or (WIDTH // 2 + 5 - 5 <= car.x <= WIDTH // 2 + 5 + 5 and HEIGHT // 2 - 300 <= car.y <= HEIGHT // 2 - 190)
        ):
            collision = True

        green_zone = pygame.Rect(WIDTH // 2 - 5, HEIGHT // 2 - 300, 10, 110)

        if car_rect.colliderect(green_zone):
            reward = 1000
            done = True
            print("Case finale atteinte")
        elif collision:
            reward = -500
            pygame.draw.circle(screen, RED, (50, 50), 10), screen.blit(pygame.font.Font(None, 24).render(f"{reward}", True, BLACK), (65, 45))
            done = True
        else:
            reward = 10
            done = False
        
        # pénaliser si un capteur gauche ou droit est plus grand que le milieu
        if car.sensors[0] > car.sensors[1] or car.sensors[2] > car.sensors[1]:
            reward = -10
            pygame.draw.circle(screen, RED, (50, 50), 10), screen.blit(pygame.font.Font(None, 24).render(f"{reward}", True, BLACK), (65, 45))
            done = False

        next_state = car.sense()
        agent.update_q_table(state, action, reward, next_state, done)

        score += reward

        if done:
            # mettre à jour le meilleur score si nécessaire
            if score > best_score:
                best_score = score
                print(f"Nouveau meilleur score : {best_score}")

            car.reset()
            print(f"Épisode {episode} terminé avec un score de {score}. Meilleur score: {best_score}")
            episode += 1
            score = 0

            # sauvegarder les scores et le nombre d'épisodes
            save_scores(best_score, episode)

            # reduire EPSILON pour l'exploration
            EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        draw_text(screen, f"Épisode: {episode}", 24, BLACK, 10, HEIGHT - 100)
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
