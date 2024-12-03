import pygame
import os
from car import *
from q_learning import *

from utils import (
    generate_random_obstacles,
    draw_obstacles,
    load_scores,
    save_scores,
    draw_text,
    draw_reward,
)

pygame.init()

# Constantes
WIDTH, HEIGHT = 800, 800
FPS = 60
WHITE = (255, 255, 255)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Initialisation de l'écran
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q Learning")
background_image = pygame.image.load("circuit.png")

# Q-table
Q_TABLE_FILE = "q_table.pkl"

# Obstacles
NUM_OBSTACLES = 4
OBSTACLES = generate_random_obstacles(NUM_OBSTACLES, radius=250, min_spacing=100, min_start_distance=100)


def main():
    global EPSILON, OBSTACLES
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
        reward = 0
        
        screen.fill(WHITE)
        screen.blit(background_image, (0, 0))
        #draw_track()
        draw_obstacles(screen, OBSTACLES)


        car.draw()
        car.draw_sensors()

        state = car.sense(OBSTACLES)
        action = agent.choose_action(state)
        car.move(action)


        # detection de collision
        collision = False
        car_rect = pygame.Rect(car.x - 15, car.y - 15, 30, 30)
        done = False
        # detection colisions 
        distance_from_center_squared = (car.x - WIDTH // 2) ** 2 + (car.y - HEIGHT // 2) ** 2
        if not (200 ** 2 <= distance_from_center_squared <= 300 ** 2):
            collision = True
        elif WIDTH // 2 - 10 <= car.x <= WIDTH // 2 and HEIGHT // 2 - 300 <= car.y <= HEIGHT // 2 - 190:
            collision = True

        for obstacle in OBSTACLES:
            if obstacle.collidepoint(car.x, car.y):  # Vérifie si le centre de la voiture touche l'obstacle
                collision = True


        

        green_zone = pygame.Rect(WIDTH // 2 + 2, HEIGHT // 2 - 300, 10, 110)
        black_zone = pygame.Rect(WIDTH // 2 - 10, HEIGHT // 2 - 300, 10, 110)
        #pygame.draw.rect(screen, BLUE, black_zone, 3)  # Dessine la case finale en jaune
        #pygame.draw.rect(screen, (255, 255, 0), green_zone, 3)  # Dessine la case finale en jaune


        # gestion reward
        if car_rect.colliderect(green_zone):
            reward = 1000
            done = True
            print("Case finale atteinte")


        elif collision:
            reward = -100
            done = True

        # gestion actions a risque
        if not done: 
            if car.sensors[0] < car.sensors[2]:
                reward = -8 if action in {0, 1} else 10
            elif car.sensors[2] < car.sensors[0]:
                reward = -8 if action in {2, 1} else 10

        if car.sensors[1]:
            if action == 1:
                reward = -200
            else:
                reward = 100

        next_state = car.sense(OBSTACLES)
        agent.update_q_table(state, action, reward, next_state, done)
        draw_reward(screen, reward, RED)
        score += reward

        if done:
            OBSTACLES = generate_random_obstacles(NUM_OBSTACLES)

            # mettre à jour le meilleur score si nécessaire
            if score > best_score:
                best_score = score

            car.reset()
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
        draw_text(screen, f"Alpha: {ALPHA:.2f}", 24, BLACK, 10, HEIGHT - 20)

        pygame.display.flip()
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()


if __name__ == "__main__":
    main()