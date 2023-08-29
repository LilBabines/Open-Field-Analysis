
<p align="center">
<img src="https://tonic.inserm.fr/wp-content/uploads/2019/10/Tonic-bandeau-home2_1940sur290.jpg" width="95%">
</p>
<div align="center">
</div>

# Vision Artificielle pour l’Analyse de Comportement Animal

## Projet de Stage (en développement):

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
- Set Up Cuda/cuDNN (c'est mieux mais pas obligatoire, je l'utilise pour la détection [PyToch] et pour la sychronisation [CuPy] (version cuda de numpy) ) (Donc si pas de GPU, remplacer Cupy par numpy, thats it normalement)

### Anaconda promt :



1.  placer vous à l'endroit ou le projet sera construit (commande "cd")

  
2. >git clone https://github.com/LilBabines/Open-Field-Analysis.git

3. >cd Open-Field-Analysis/

4. >conda create --name OPENFIELD python=3.9 

5. >conda activate OPENFIELD

6. >pip install -r ./requirements.txt

7. >cd ./model  

8. >git clone https://github.com/ultralytics/yolov5  # clone yolov5 repository

9.  >cd yolov5

10. >pip install -r requirements.txt 


L'environnement est normalement bien instalé.
Testons le :

1. placer une video dans le projet (à la racine si vous voulez)
2. ouvrir Anaconda PowerShell Prompt
3. >cd path_to_the_project/Open_Field_Analysis
4. >python .\sample\seg_rat\segmentation_yolo.py --source video_path (./video1.MP4,   si video à la racine)
   
Le code se lance, charge le reseau de neurones, puis la détection du rat et des redressement ce fait directement sur l'image.
Pour quitter les fenêtres de lecture de video : cliquer sur la video puis sur la touche **a** ou/et **echap**.

Si vous avez une erreur de package style : *No module named 'module_name'*. C'est qu'il faut l'installer séparemment (il manque au requirement.txt). Trouver le package sur https://pypi.org/ (premier site google search souvent). Pour l'installer assurez vous d'être dans le bon environnment conda, celui créé plus tôt (*OPENFIELD* si pas changé). 


## Architecture à respecter :

```md
OpenField
├── sample (code source)
├── utils (code source annexe peu utile, à concaténer avec sample)
├── model (dossier des poids yolov5 et le répertoire de yolov5)
├── cfg (dossier des configurations)
    ├──config.xlsx (tableau qui répertarie les données)
    └──run_cfg.yaml (constante du projet, chemin de sorti, emplacement des poids, frame rate, etc.)
├── analyse (dossier résultats, métriques et autre comptage )
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
Les fichiers sont très importants. Le fichier *config.xlsx* doit contenir toutes les information necessaires sur les données. Pour que les vidéos ne soient ni mélangées ni manquantes en autre. Chaque ligne corresponds à une expérience et doit contenire : nom rat, nom expéreience (correspondant aux dossier associé), nom des videos (sans .MP4), start_time (vaux zéro si pas spécifié), minute:seconde de l'arrivé de la vitre, anomalie des video : 'son' si le fichier audio est détaché.

Le fichier *run_cfg.yaml*, contient les liens, les paramètre du projet, toutes les constances. Le nom des variables sont le plus explicite possible. *SAVE_PATH_TRAJECTORY* et *SAVE_PATH_PAW* sont les chemins ont seront enregistrer les sorties csv des algo de l'extraction de la trajectoire et de la détectoin d'appui. *DATA_PATH* le répertoire racine des donnés, il peut être placé en réalité (normalement) n'importe où, il suffit simplement bien le marquer sur le fichier .yaml, *DATA_CONFIG* emplacement de la configuratoin des données (peut changer en fonction des données qui veulent être traitées, ne mettre dans le tableau seulement les expériences qui veulent être analysées), *ANALYSE_PATH* répertoire où seront enregistrée les analyse des csv, *SIDE_LESION* répertorie la latéralité des membre handicapé par la lésoin pour chaque rat, *FRAME_RATE* intervalle frame de décsion (garder 1 le plus possible pour la détection d'appui notamment, si il vient à changé attention d'adapter les décisions).



## Utilisation :
Dans un premier temps faire le fichier *confg.xlsx*.

Puis terminal command anaconda (toujours se placer dans le répertoire OpenField pour terminal, sinon ajuster les chemin des fichiers et être dans le bon environnement) : 
1. Test un modèle sur une video
   
    - >python .\sample\seg_rat\segmentation_yolo.py --source video_path
    - >python .\sample\seg_rat\segmentation_yolo.py --source video_path --weights weights_path
2. Pour traiter les vidéos
   1. trajectoire :
        - > python .\sample\trajectory\main.py --mode test/run
    2. appui :
        - >python .\sample\paw_detection\main_paw.py --mode run/test
3. Analyse résultat
   1. Trajectoire :
       - >python .\sample\output_read\out_read.py --mode trajectoire/appui 

Le répertoire YOLOv5 créé durant l'instalation permet de nombreuses choses (entrainement, détectoin, analyse, seuil, score F1 et autre) voir   https://github.com/ultralytics/yolov5. Notamment pour enregistrer une video de prédiction ou filtrer les prediction avec leurs scores de confiance, il faudrait utiliser le fichier .\model\yolov5\detect.py. Le fichier comporte des explicatoin d'utilisation au dbut. Conseil d'utilisatoin :
- >python .\model\yolov5\detect.py --weights [path_weights] --source [vdieo_path]
- > python .\model\yolov5\detect.py --weights .\model\poids\LAST_LAST_LAST.pt --source .\data\test_rat\test_exp\GOPRO1\****.MP4 

- >python .\model\yolov5\detect.py --weights .\model\poids\best_1920.pt --source .\data\test_rat\test_exp\GOPRO1\****.MP4 --img-size 1920 --conf-thres 0.6 --line-thickness 5

