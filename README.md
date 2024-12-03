# Q Learning Car Simulation

Ce projet implémente une simulation de voiture utilisant l'apprentissage par renforcement Q-learning pour naviguer sur une piste. Le but est d'entraîner une voiture à éviter les obstacles et à atteindre une zone cible, en utilisant des capteurs et en optimisant ses mouvements.

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
python main.py
```

## Description du projet

Ce projet utilise **Pygame** pour créer une simulation de conduite où une voiture est formée à naviguer sur un parcours via un algorithme de **Q-learning**. La voiture est contrôlée par un agent qui choisit entre trois actions : tourner à gauche, aller tout droit ou tourner à droite. La récompense est attribuée selon la distance parcourue, la capacité de la voiture à éviter les collisions et à atteindre la zone finale (une case verte).

### Paramètres d'apprentissage

- **Alpha (ALPHA)** : Taux d'apprentissage (remplacement des anciennes informations par de nouvelles).
- **Gamma (GAMMA)** : Facteur de récompense future (importance des futures récompenses).
- **Epsilon (EPSILON)** : Contrôle l'exploration (probabilité de choisir une action aléatoire).
- **EPSILON_DECAY** : Taux de décroissance d'épsilon pour réduire progressivement l'exploration.

### Actions possibles

- **0** : Tourner à gauche.
- **1** : Aller tout droit.
- **2** : Tourner à droite.

### Objectif

L'objectif est de conduire la voiture jusqu'à une zone spécifique en évitant les collisions avec les obstacles. La voiture est équipée de capteurs qui détectent les distances jusqu'aux obstacles environnants.

Pendant la simulation, l'agent de Q-learning choisit les actions optimales pour maximiser les récompenses, permettant ainsi à la voiture d'atteindre la zone verte tout en évitant les collisions.

## Sauvegarde de la table Q

La table Q est sauvegardée dans un fichier nommé **q_table.pkl**. Si le fichier existe, la table Q est chargée au démarrage du programme, permettant de continuer l'apprentissage à partir d'une session précédente.

## Structure des fichiers

- **main.py** : Script principal pour lancer la simulation.
- **q_learning.py** : Implémentation de l'algorithme Q-learning.
- **car.py** : Définition de la classe Car, incluant les capteurs et les mouvements.
- **utils.py** : Fonctions utilitaires pour gérer les obstacles, les scores et l'affichage.
- **circuit.png** : Image du circuit utilisé comme arène pour la simulation.
- **requirements.txt** : Liste des dépendances Python nécessaires.

