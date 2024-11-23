# Q Learning Car Simulation

Ce projet implémente une simulation de voiture utilisant l'apprentissage par renforcement Q-learning pour naviguer sur une piste. Le but est d'entraîner une voiture à éviter les obstacles et atteindre une zone cible, en utilisant des capteurs et en optimisant ses mouvements.

## Commandes pour commencer le projet

Pour commencer avec ce projet, suivez les étapes ci-dessous :

1. Clonez le dépôt :

```sh
git clone https://github.com/aymane8414/Q_Learning.git
```

2. Accédez au répertoire du projet :

```sh
cd Q_Learning
```

3. Installez les dépendances :

```sh
pip install -r requirements.txt
```

4. Lancez la simulation :

```sh
python q_learning.py
```

## Description du projet

Ce projet utilise **Pygame** pour créer une simulation de conduite où une voiture est formée à naviguer sur un parcours via un algorithme de **Q-learning**. La voiture est contrôlée par un agent qui choisit entre trois actions : tourner à gauche, aller tout droit, ou tourner à droite. La récompense est donnée selon la distance parcourue et la capacité de la voiture à atteindre la zone finale (une case jaune).

### Paramètres d'apprentissage

- **Alpha (ALPHA)** : Taux d'apprentissage (à quel point les nouvelles informations remplacent les anciennes).
- **Gamma (GAMMA)** : Facteur de récompense future (importance des futures récompenses).
- **Epsilon (EPSILON)** : Contrôle l'exploration (probabilité de choisir une action aléatoire).
- **EPSILON_DECAY** : Taux de décroissance d'épsilon pour réduire progressivement l'exploration.

### Actions possibles

- **0** : Tourner à gauche.
- **1** : Aller tout droit.
- **2** : Tourner à droite.

### Objectif

L'objectif est de conduire la voiture jusqu'à une zone spécifique en évitant les collisions avec les murs. La voiture est dotée de capteurs qui détectent les distances jusqu'aux obstacles (murs).

Pendant la simulation, la voiture se déplace en fonction des actions choisies par l'agent Q-learning. L'objectif est d'atteindre la zone jaune tout en évitant les obstacles.

## Sauvegarde de la table Q

La table Q est sauvegardée dans un fichier nommé **q_table.pkl**. Si le fichier existe, la table Q est chargée à partir de celui-ci au démarrage du programme, permettant de continuer l'apprentissage à partir d'une précédente session.

