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
longueur_cuve = 10 #axe x
largeur_cuve = 4 #axe y
epaisseur_cuve = 2 #axe z
N = 15
centre_y = largeur_cuve // 2 #on repère le centre de la zon

viscosite = 0.005
tau = 3 * viscosite + 0.5
omega = 1. / tau
Re = 1 #Nombre de Reynolds
#Position cylindre
xc = 5
yc = largeur_cuve // 2
zc = epaisseur_cuve // 2
rc = 1
#Vitesse pour le flow
#calcul recurrents
u0 = 0.1
u02 = u0 ** 2
poids = np.array([2 / 9] + [1 / 9] * 6 + [1 / 72] * 8) #poids de D3Q15
cs2 = 1 #paramètre de vitesse de la maille
# nu = cs2 * (tau + 0.5)
nu = 1
nu2 = nu ** 2

vitesses = np.array([
	[0, 0, 0], #Rest -> 0
	[1, 0, 0], #+x -> 1
	[-1, 0, 0], #-x -> 2
	[0, 1, 0], #+y -> 3
	[0, -1, 0], #-y -> 4
	[0, 0, 1], #z ->5,
	[0, 0, -1], #-z ->6,
	[1, 1, 1], #+x+y+z -> 7
	[-1, 1, 1], #-x+y+z -> 8
	[1, -1, -1], #+x-y-z -> 9
	[-1, -1, -1], #-x-y-z -> 10
	[1, -1, 1], #+x-y+z -> 11
	[-1, 1, -1], #-x+y-z -> 12
	[1, 1, -1], #+x+y-z -> 13
	[-1, -1, 1], #-x-y+z -> 14
])
#vecteurs vitesse vi
#Les vecteurs sont organises telle que la somme de leurs coordonnees
#donne la position dans le tableau de leur oppose


def zoneCylindre(x, y, z):
	""" Retourne True si on est dans la zone du cylindre """
	global zc, yx, xc, rc
	d = (z - zc) ** 2 + (y - yc) ** 2 - rc ** 2
	return d >= 0 and x <= xc

def diff(u, i, j):
	"""Retourne duidj - dujdi"""
	duidj = (np.roll(u[i], -1, axis = j) - np.roll(u[i], 1, axis = j))
	dujdi = (np.roll(u[j], -1, axis = i) - np.roll(u[j], 1, axis = i))
	return duidj - dujdi

def calculVorticite(u):
	""" Autre calcul de vorticite """
	terme_x = diff(u, 2, 0)
	terme_y = diff(u, 1, 2)
	terme_z = diff(u, 0, 1)
	resultat = np.array([terme_y, terme_y, terme_z])
	return resultat

def propage_un(i, f):
	global vitesses
	""" Sert à propager une direction """

	dx, dy, dz = vitesses[i]
	#on fait la propagation
	f[i] = np.roll(f[i], dy, axis = 0)
	f[i] = np.roll(f[i], dx, axis = 1)
	f[i] = np.roll(f[i], dz, axis = 2)

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
		autre = i + np.sum(vitesses)
		print(i)
		print(autre)
		f[i][barrier[i]] = f[autre][barrier[0]]



def collisions():
	""" Cette fonction gere le terme de collisions """
	global f, ux, uy, uz, rho, u0, u02, omega

	#Valeurs macroscopiques
	rho = np.sum(f, axis = 0)
	ux = np.zeros((largeur_cuve, longueur_cuve, epaisseur_cuve))
	uy = np.zeros((largeur_cuve, longueur_cuve, epaisseur_cuve))
	uz = np.zeros((largeur_cuve, longueur_cuve, epaisseur_cuve))

	for i in range(N):
		ux += vitesses[i, 0] * f[i]
		uy += vitesses[i, 1] * f[i]
		uz += vitesses[i, 2] * f[i]

	ux = ux / rho
	uy = uy / rho
	uz = uz / rho
	terme1 = 1 - 1.5 * (ux ** 2 + uy ** 2 + uz ** 2)

	#on applique la formule BGK
	for i in range(N):
		ps = vitesses[i, 0] * ux + vitesses[i, 1] * uy + vitesses[i, 2] * uz
		terme2 = 3 * ps / nu
		terme3 = 4.5 * np.power(ps, 2) / nu2
		f[i] = (1 - omega) * f[i] + omega * poids[i] * rho * (terme1 + terme2 + terme3)
		# try:
		# 	f[i] = (1 - omega) * f[i] + omega * poids[i] * rho * (terme1 + terme2 + terme3)
		# except RuntimeWarning:
		# 	print('hihi')
	
	#On force le flow en entree
	terme1 = 1 - 1.5 * u02 / nu2

	for i in [1, 2, 7, 8, 9, 10, 11, 12, 13, 14]:
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
	barrier = np.zeros((N, largeur_cuve, longueur_cuve, epaisseur_cuve), dtype = bool)
	#barriere horizontale
	# barrier[0, 0, :] = True
	# barrier[0, largeur_cuve - 1, :] = True
	# on construit la barriere cylindrique
	for x in range(longueur_cuve):
		for y in range(largeur_cuve):
			for z in range(epaisseur_cuve):
				if zoneCylindre(x, y, z):
					barrier[0, y, x, z] = True
	# barrier[0, (largeur_cuve // 2) - taille_barriere:(largeur_cuve // 2) + taille_barriere, largeur_cuve // 2] = True
	# barrier[0, :hauteur_barriere, :largeur_barriere] = True
	# barrier[0, largeur_cuve - hauteur_barriere:, :largeur_barriere] = True

	#Ces booleens servent a gerer les collisions
	for i in range(N):
		dx, dy, dz = vitesses[i]
		#On calcule les points aux limites des barrieres
		barrier[i] = np.roll(barrier[0], -1 * dy, axis = 0) #-1 car l'axe y numpy est inverse
		barrier[i] = np.roll(barrier[i], dx, axis = 1)
		barrier[i] = np.roll(barrier[i], dz, axis = 2)

	##########On initialise avec un flux constant
	f = np.ones((N, largeur_cuve, longueur_cuve, epaisseur_cuve))
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
	ux = np.zeros((largeur_cuve, longueur_cuve, epaisseur_cuve))
	uy = np.zeros((largeur_cuve, longueur_cuve, epaisseur_cuve))
	uz = np.zeros((largeur_cuve, longueur_cuve, epaisseur_cuve))

	for i in range(N):
		ux += vitesses[i, 0] * f[i]
		uy += vitesses[i, 1] * f[i]
		uz += vitesses[i, 2] * f[i]

	ux = ux / rho
	uy = uy / rho
	uz = uz / rho

	##################################
	########## Affichage #############
	##################################
	print('Affichage')
	# fig, axes = plt.subplots(1, 2, figsize = (16, 6))
	fig, axes = plt.subplots(1, 1)
	# ax1, ax2 = axes
	ax1 = axes
	
	w = calculVorticite(np.array([ux, uy, uz]))
	# Affichage vorticite
	fluidImage = ax1.imshow(
		w[2, :, :, zc], 
		origin='lower', 
		norm = plt.Normalize(-.1,.1), 
		cmap = plt.get_cmap('jet'), 
		interpolation = 'none'
	)

	#affichage des barrieres
	bImageArray = np.zeros((largeur_cuve, longueur_cuve, 4), np.uint8)	# an RGBA image
	bImageArray[barrier[0, :, :, zc], 3] = 255								# set alpha=255 only at barrier sites
	barrierImage = ax1.imshow(bImageArray, origin = 'lower', interpolation='none')

	#Affichage du flow
	# X = np.linspace(0, longueur_cuve, longueur_cuve)
	# Y = np.linspace(0, largeur_cuve, largeur_cuve)
	# strm = ax2.streamplot(X, Y, ux, uy, linewidth = 2, cmap = plt.cm.autumn)

	#fonction d'animation
	def nextFrame(k, *args):
		global u0, ux, uy, uz, barrier

		for step in range(30): #A ajuster pour que ça soit bien
			propagation()
			collisions()
		w = calculVorticite(np.array([ux, uy, uz]))
		fluidImage.set_array(w[2, :, :, zc])

		# if k > 0:
		# 	u0 = 0

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