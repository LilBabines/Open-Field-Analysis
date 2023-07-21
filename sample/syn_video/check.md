# Synchronisation Video

## **Contexte**

Le rat est filmé sous 4 angles différents avec 4 caméras filmant la même scène. Pour recroisser les résulats de chaque caméra il faut faire correspondre les 4 frames.

## **Méthode**

Afin de synchroniser les 4 videos, on passe par l'analyse du son.
Les audios sont extrait des fichiers MP4, echantillonnage 44,1 kH
On calcule 3 corrélation entre les 4 bandes sonnore :

- Corrélatoin 1 : audio2 / audio1
- ------------  2 : audio3 / audio2
- ------------ 3 : aufio4 / audio3

Après chaque corrélation :

- Calcul de l'argmax puis du délai
- en fonction du signe on avance la première ou la deuxième video

On se retrouve donc à recaler toute le video sur celle qui est le plus en retard ie allumer en dernière.

En pratique les caméras sont lancées dans l'ordre : 1, 2, 3 puis 4 ( voir 1, 2, 4, 3 par moment). 

On remarque un décalage de 0 à 2 frames soit environ  $0.03$ secondes ( video en 60 FPS).

## **Problèmes** :

- si l'on test le recalage à partir de la frame zéro, la dernière video est complètement décalée, alors que si l'on commence à partir d'une centaine de frame la test est parfait. Hypothèse : les frames corrompues en début de video décale la dernière video, ales autres video elles n'ont pas le problème car lerecalage passe les frames corrompues.
- Quand on retourne une video mal enregistrer seul la video est enregistrer, OpenCV ne prend pas en compte l'audio ( ce qui esrt tout a fait normal) il faut donc penser à enregistrer l'audio de la video retournée puis enregistrer sonhomologue dans le bon sens.

