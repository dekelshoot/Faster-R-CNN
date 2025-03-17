# Projet Faster R-CNN

Ce projet utilise le modèle Faster R-CNN pour la détection d'objets sur le jeu de données Pascal VOC 2012.

## Description

Faster R-CNN (Region-based Convolutional Neural Networks) est un modèle de détection d'objets qui combine la génération de propositions régionales et la détection d'objets en une seule étape. Il utilise un réseau de propositions régionales (RPN) pour générer des propositions d'objets et un réseau de détection pour classifier ces propositions et affiner leurs coordonnées. Ce projet implémente ce modèle pour détecter et classifier des objets dans des images, en utilisant le jeu de données Pascal VOC 2012, qui est un standard pour les tâches de détection d'objets.

## Prérequis

- **Python 3.7 ou supérieur** : Le langage de programmation utilisé pour ce projet.
- **pip** : Le gestionnaire de paquets Python, utilisé pour installer les dépendances.
- **git** : Utilisé pour cloner le dépôt du projet.

## Installation

1. **Clonez le dépôt :**

   Clonez le dépôt GitHub du projet sur votre machine locale et accédez au répertoire du projet.

   ```bash
   git clone https://github.com/dekelshoot/Faster-R-CNN.git
   cd faster_rcnn
   ```

2. **Créez un environnement virtuel :**

   Créez un environnement virtuel pour isoler les dépendances du projet. Cela permet de s'assurer que les dépendances du projet n'interfèrent pas avec d'autres projets ou installations Python sur votre machine.

   ```bash
   python -m venv venv
   ```

3. **Activez l'environnement virtuel :**

   Activez l'environnement virtuel que vous venez de créer. Cela vous permet d'utiliser les dépendances installées dans cet environnement.

   - Sur Windows :
     ```bash
     venv\Scripts\activate
     ```
   - Sur macOS/Linux :
     ```bash
     source venv/bin/activate
     ```

4. **Installez les dépendances :**

   Installez toutes les dépendances nécessaires à partir du fichier `requirements.txt`. Ce fichier contient une liste de toutes les bibliothèques Python dont le projet a besoin.

   ```bash
   pip install -r requirements.txt
   ```

5. **Téléchargez les données :**

   Téléchargez le jeu de données Pascal VOC 2012 en exécutant le script `download_data.sh`. Ce script automatisera le processus de téléchargement des données nécessaires pour entraîner et tester le modèle.

   ```bash
   bash download_data.sh
   ```

6. **Téléchargez les modules nécessaires à l'entraînement :**

   Téléchargez les modules supplémentaires nécessaires pour l'entraînement du modèle en exécutant le script `module_for_model_training.sh`. Ces modules peuvent inclure des poids pré-entraînés ou d'autres ressources nécessaires pour l'entraînement.

   ```bash
   bash module_for_model_training.sh
   ```

## Utilisation

1. **Entraînez et sauvegardez le modèle :**

   Entraînez le modèle sur le jeu de données Pascal VOC 2012 en exécutant le script `training_model.py`. Ce script entraînera le modèle et sauvegardera le modèle entraîné dans un fichier.

   ```bash
   python training_model.py
   ```

2. **Testez le modèle sur une image :**

   Testez le modèle sur une image en exécutant le script `test.py` avec le chemin de l'image que vous souhaitez tester. Le script affichera l'image avec les objets détectés et leurs classes.

   ```bash
   python test.py <chemin_vers_l_image>
   ```

   Remplacez `<chemin_vers_l_image>` par le chemin de l'image que vous souhaitez tester.

## Structure du projet

- `download_data.sh` : Script pour télécharger le jeu de données Pascal VOC 2012.
- `module_for_model_training.sh` : Script pour télécharger les modules nécessaires à l'entraînement du modèle.
- `training_model.py` : Script pour entraîner le modèle.
- `test.py` : Script pour tester le modèle sur une image.
- `requirements.txt` : Fichier listant les dépendances nécessaires pour le projet.

## Contribuer

Les contributions sont les bienvenues ! Veuillez soumettre une pull request ou ouvrir une issue pour discuter des changements que vous souhaitez apporter.

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.