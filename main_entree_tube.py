#!/usr/bin/env
# coding: utf8
#################### Bibliothèques ###############

import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
# plt.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\bin\\ffmpeg.exe"
import matplotlib.animation as animation
from copy import deepcopy
from time import time

import warnings
warnings.filterwarnings("error")

# from multiprocessing import Pool

print('Librairies importees')

################### Parametres ###################
#Parametres spatiaux
longueur_cuve = 100 #axe x
largeur_cuve = 40 #axe y
N = 9
centre_y = largeur_cuve / 2 #on repère le centre de la zon

viscosite = 0.005
tau = 3 * viscosite + 0.5
omega = 1. / tau
Re = 1 #Nombre de Reynolds
#Position cylindre
xc = longueur_cuve // 4
yc = largeur_cuve // 2
rc = 3
#Vitesse pour le flow
#calcul recurrents
u0 = 0.1
u02 = u0 ** 2
poids = np.array([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36]) #poids de D2Q9
cs2 = 1 #paramètre de vitesse de la maille
# nu = cs2 * (tau + 0.5)
nu = 1
nu2 = nu ** 2

vitesses = np.array([
	[0, 0], #Rest -> 0
	[1, 0], #E -> 1
	[1, 1], #NE -> 2
	[0, 1], #N -> 3
	[-1, 1], #NW -> 4
	[-1, 0], #W -> 5
	[-1, -1], #SW -> 6
	[0, -1], #S -> 7
	[1, -1]]) #SE -> 8
#vecteurs vitesse vi


def zoneCylindre(x, y):
	""" Retourne True si on est dans la zone du cylindre """
	global xc, yx, rc
	d = (x - xc) ** 2 + (y - yc) ** 2 - rc ** 2
	return d <= 0

def calculVorticite(ux, uy):
	""" Autre calcul de vorticite """
	duydx = (np.roll(uy, -1, axis = 1) - np.roll(uy, 1, axis = 1))
	duxdy = (np.roll(ux, -1, axis = 0) - np.roll(ux, 1, axis = 0))
	w = duydx - duxdy
	return w

def propage_un(i, f):
	global vitesses
	""" Sert à propager une direction """

	est_ouest, nord_sud = vitesses[i]
	#on fait la propagation
	f[i] = np.roll(f[i], nord_sud, axis = 0)
	f[i] = np.roll(f[i], est_ouest, axis = 1)

	return f[i]

def propage_un_parralel(args):
	return propage_un(*args)

# propage_un_parralel.parallel = parallel_attribute(propage_un_parralel)

def propagation():
	""" On propage les densites selon les directions """
	global f
	#Valuer macroscopique
	rho = np.sum(f, axis = 0)
	#Propagation
	for i in range(N):
		f[i] = propage_un(i, f)
	# f = propage_un_parralel.parallel(N)
	# from multiprocessing import Pool
	# pool = Pool()
	# results = [pool.apply(propage_un, args = (i, f)) for i in range(N)]
	# f = np.array(results)
	# pool.close()
	# pool.join()

	#collision avec les obstacles
	for i in range(1, N):
		#On recupere la direction "opposee"
		autre = (i + 4) % 8
		#petit hack, l'oppose de 4 c'est 8 mais le modulo renvoie 0
		if autre == 0:
			autre = 8
		#si on se trouve juste apres barriere, on échange les directions
		f[i][barrier[i]] = f[autre][barrier[0]]



def collisions():
	""" Cette fonction gere le terme de collisions """
	global f, ux, uy, rho, u0, u02, omega

	#Valeurs macroscopiques
	rho = np.sum(f, axis = 0)
	ux = np.zeros((largeur_cuve, longueur_cuve))
	uy = np.zeros((largeur_cuve, longueur_cuve))

	for i in range(N):
		ux += vitesses[i, 0] * f[i]
		uy += vitesses[i, 1] * f[i]

	ux = ux / rho
	uy = uy / rho
	terme1 = 1 - 1.5 * (ux ** 2 + uy ** 2)

	#on applique la formule BGK
	for i in range(N):
		ps = vitesses[i, 0] * ux + vitesses[i, 1] * uy
		terme2 = 3 * ps / nu
		terme3 = 4.5 * np.power(ps, 2) / nu2
		f[i] = (1 - omega) * f[i] + omega * poids[i] * rho * (terme1 + terme2 + terme3)
		# try:
		# 	f[i] = (1 - omega) * f[i] + omega * poids[i] * rho * (terme1 + terme2 + terme3)
		# except RuntimeWarning:
		# 	print('hihi')
	
	#On force le flow en entree
	terme1 = 1 - 1.5 * u02 / nu2

	for i in [1, 2, 4, 5, 6, 8]:
		ps = vitesses[i, 0] * u0
		terme2 = 3 * ps / nu
		terme3 = 4.5 * np.abs(vitesses[i, 0]) * u02 / nu2

		f[i, :, 0] = poids[i] * (terme1 + terme2 + terme3)

	# for i in [1, 3, 4, 5, 7, 8]:
	# 	ps = vitesses[i, 0] * u0
	# 	terme2 = 3 * ps / nu
	# 	terme3 = 4.5 * np.abs(vitesses[i, 0]) * u02 / nu2

	# 	f[i, 0, :] = poids[i] * (terme1 + terme2 + terme3)
	# 	f[i, largeur_cuve - 1, :] = poids[i] * (terme1 + terme2 + terme3)


if __name__ == '__main__':
	########On definit la barriere
	barrier = np.zeros((9, largeur_cuve, longueur_cuve), dtype = bool)
	#barriere horizontale
	# barrier[0, 0, :] = True
	# barrier[0, largeur_cuve - 1, :] = True
	# on construit la barriere cylindrique
	# for x in range(longueur_cuve):
	# 	for y in range(largeur_cuve):
	# 		if zoneCylindre(x, y):
	# 			barrier[0, y, x] = True
	hauteur_barriere = 15
	largeur_barriere = 5
	# barrier[0, (largeur_cuve // 2) - taille_barriere:(largeur_cuve // 2) + taille_barriere, largeur_cuve // 2] = True
	barrier[0, :hauteur_barriere, :largeur_barriere] = True
	barrier[0, largeur_cuve - hauteur_barriere:, :largeur_barriere] = True

	#Ces booleens servent a gerer les collisions
	for i in range(N):
		est_ouest, nord_sud = vitesses[i]
		#On calcule les points aux limites des barrieres
		barrier[i] = np.roll(barrier[0], -1 * nord_sud, axis = 0) #-1 car l'axe y numpy est inverse
		barrier[i] = np.roll(barrier[i], est_ouest, axis = 1)

	##########On initialise avec un flux constant
	f = np.ones((N, largeur_cuve, longueur_cuve))
	terme1 = 1 - 1.5 * u02 / nu2
	for i in range(N):
		#produit scalaire de la vitesse forcee avec la vitesse horizontale
		ps = vitesses[i, 0] * u0
		terme2 = 3 * ps / nu
		terme3 = 4.5 * np.abs(vitesses[i, 0]) * u02 / nu2

		f[i] = poids[i] * (terme1 + terme2 + terme3)
	#Valeurs macroscopiques
	rho = np.sum(f, axis = 0) #densite
	#vitesses
	ux = np.zeros((largeur_cuve, longueur_cuve))
	uy = np.zeros((largeur_cuve, longueur_cuve))

	for i in range(N):
		ux += vitesses[i, 0] * f[i]
		uy += vitesses[i, 1] * f[i]

	ux = ux / rho
	uy = uy / rho

	##################################
	########## Affichage #############
	##################################
	print('Affichage')
	# fig, axes = plt.subplots(1, 2, figsize = (16, 6))
	fig, axes = plt.subplots(1, 1)
	# ax1, ax2 = axes
	ax1 = axes
		
	# Affichage vorticite
	fluidImage = ax1.imshow(
		calculVorticite(ux, uy), 
		origin='lower', 
		norm = plt.Normalize(-.1,.1), 
		cmap = plt.get_cmap('jet'), 
		interpolation = 'none'
	)

	#affichage des barrieres
	bImageArray = np.zeros((largeur_cuve, longueur_cuve, 4), np.uint8)	# an RGBA image
	bImageArray[barrier[0], 3] = 255								# set alpha=255 only at barrier sites
	barrierImage = ax1.imshow(bImageArray, origin = 'lower', interpolation='none')

	#Affichage du flow
	# X = np.linspace(0, longueur_cuve, longueur_cuve)
	# Y = np.linspace(0, largeur_cuve, largeur_cuve)
	# strm = ax2.streamplot(X, Y, ux, uy, linewidth = 2, cmap = plt.cm.autumn)

	#fonction d'animation
	def nextFrame(k, *args):
		global u0, ux, uy, barrier

		for step in range(30): #A ajuster pour que ça soit bien
			propagation()
			collisions()

		fluidImage.set_array(calculVorticite(ux, uy))

		if k > 0:
			u0 = 0

		# ax2.cla()
		# strm = ax2.streamplot(X, Y, ux, uy, linewidth = 2, cmap = plt.cm.autumn)
		# plt.draw()
		return (fluidImage, barrierImage)

	# Set up formatting for the movie files
	# Writer = animation.writers['ffmpeg']
	# writer = Writer(fps = 15, metadata = dict(artist = 'Me'), bitrate = 1800)

	#animation
	ani = animation.FuncAnimation(
		fig, 
		nextFrame, 
		frames = range(100), 
		repeat = False, 
		interval = 1, 
		blit = True)

	# ani.save('test.mp4', writer = writer)
	plt.show()