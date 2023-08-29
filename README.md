
<p align="center">
<img src="https://tonic.inserm.fr/wp-content/uploads/2019/10/Tonic-bandeau-home2_1940sur290.jpg" width="95%">
</p>
<div align="center">
</div>

# Vision Artificielle pour l’Analyse de Comportement Animal

## Projet de Stage :

**Université** : [Toulouse - UT3 Paul Sabatier](https://www.univ-tlse3.fr/)

**Lieu du Stage** : [INSER - ToNIC](https://tonic.inserm.fr/)

**Auteur** : Auguste Verdier

**Tutuer** : [Thomas Pellegrini](https://www.irit.fr/~Thomas.Pellegrini/)

**Encadrants** : [Isabelle Loubinoux](https://tonic.inserm.fr/equipes/isabelle-loubinoux/), [Julien Clauzel](https://tonic.inserm.fr/equipes/julien-clauzel/), [Adrien Brilhault](https://tonic.inserm.fr/equipes/adrien-brilhault/)

**Introduction :** L'équipe iDream de l'unité ToNIC (INSERM) cherche à déveloper l'automatisation des analyses des test préclinique sur les animaux, par deeep learning principalement. Un financement de l'Université Paul Sabatier a permis  à l'équipe de lancer un projet avec de matériel et une offre de stage pour M2. Le but du stage et de construire des outils permettant d'analyser des test moteur chez le rat (test de l'Open field et test Tirage de corde).  

**Keyword** : Open-field, Object Detection, Action Recognition, YOLO, multi-views, paw dection


## Set Up :
### Requirment : 
- Install [Python](https://www.python.org/downloads/)
- Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/windows/) 
- Set Up Cuda/cuDNN (c'est mieux mais pas obligatoire) 

### Anaconda promt :

1. git clone https://github.com/LilBabines/Open-Field-Analysis.git

2. cd Open-Field-Analysis/

3. conda create --name OPENFIELD  

4. conda activate OPENFIELD

5. pip install -r ./requirements.txt

6. cd ./model  

7. git clone https://github.com/ultralytics/yolov5  # clone yolov5 repository

8. cd yolov5

9.  pip install -r requirements.txt 

L'environnement est normalement bien instalé.
Testons le : 

. . . . . . .

## Architecture à respecter :

```md
OpenField
├── sample (code source)
├── utils (code source annexe peu utile, à concaténer avec sample)
├── model (dossier des poids yolov5)
├── cfg (dossier des configurations)
    ├──config.xlsx (tableau qui répertarie les données)
    └──run_cfg.yaml (constante du projet, chemin de sorti, emplacement des poids, frame rate, etc.)
├── analyse (dossier résultats, métriques et )
└── data (dossier des données : vidéo + homography)
    └──rat1
        └──exp1
            ├──GOPRO1
                └──video_file.MP4
            ├──GOPRO2
            ├──GOPRO3
            ├──GOPRO4
            └──homography
                ├──homo1.npy
                ├──...
                └──homo4.npy
```

, chaque ligne corresponds à une expérience et doit contenire : nom rat, nom expéreience (correspondant aux dossier associé), nom des videos (sans .MP4), start_time (vaux zéro si pas spécifié), minute:seconde de l'arrivé de la vitre, anomalie des video : 'son' si le fichier audio est détaché)




