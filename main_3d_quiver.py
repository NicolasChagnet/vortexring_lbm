#!/usr/bin/env
# coding: utf8
#################### Bibliothèques ###############

import numpy as np
# import matplotlib.pyplot as plt
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
# import matplotlib.animation as animation
from copy import deepcopy
from time import time

#permet de recuperer les overflow
import warnings
warnings.filterwarnings("error")

print('Librairies importees')

################### Parametres ###################

#Parametres spatiaux
longueur_cuve = 40 #axe x
largeur_cuve = 40 #axe y
epaisseur_cuve = 20 #axe z
N = 19

#parametres du fluide
viscosite = 0.005
tau = 3 * viscosite + 0.5
omega = 1. / tau
Re = 1 #Nombre de Reynolds

#Position barriere
xc = 10
yc = largeur_cuve // 2
zc = epaisseur_cuve // 2
rc = 4

hauteur_barriere = 15
largeur_barriere = 5

#Vitesse pour le flow
u0 = 0.1
u02 = u0 ** 2
cs2 = 1 #paramètre de vitesse de la maille
# nu = cs2 * (tau + 0.5)
# nu = 1 / np.sqrt(3)
nu = 1.
nu2 = nu ** 2

#Parametres du LBM
poids = np.array([1. / 3] + [1. / 18] * 6 + [1. / 36] * 12) #poids de D3Q19
vitesses = np.array([
	[0, 0, 0], #Repos -> 0
	[0, 0, 1], #z ->1,
	[0, 0, -1], #-z ->2
	[0, 1, 0], #+y -> 3
	[0, -1, 0], #-y -> 4
	[1, 0, 0], #+x -> 5
	[-1, 0, 0], #-x -> 6
	[1, 1, 0], #+x+y -> 7
	[1, -1, 0], #+x-y -> 8
	[-1, 1, 0], #-x+y -> 9
	[-1, -1, 0], #-x-y -> 10
	[1, 0, 1], #+x+z -> 11
	[1, 0, -1], #+x-z -> 12
	[-1, 0, 1], #-x+z -> 13
	[-1, 0, -1], #-x-z -> 14
	[0, 1, 1], #+y+z -> 15
	[0, 1, -1], #+y-z -> 16
	[0, -1, 1], #-y+z -> 17
	[0, -1, -1], #-y-z -> 18
])
#vecteurs vitesse vi
#Permet de faire correspondre les vitesses en rebond
autres = [0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15]

F = np.array([0, 0, 0]) #Force de gravite
#Nulle pour l'instant, tant que ça marche pas

#Definit où est la barriere
def zoneBarriere(x, y, z):
	""" Retourne True si on est dans la zone du cylindre """
	global zc, yx, xc, rc
	d = (y - yc) ** 2 + (z - zc) ** 2 - rc ** 2
	return d >= 0 and x <= xc
	

#On l'utilise pas à cause du y pas placé au bon endroit dans f
def diff(u, i, j):
	"""Retourne duidj - dujdi"""
	duidj = (np.roll(u[i], -1, axis = j) - np.roll(u[i], 1, axis = j))
	dujdi = (np.roll(u[j], -1, axis = i) - np.roll(u[j], 1, axis = i))
	return duidj - dujdi

#marche pas pour la raison sus-mentionnee
# def calculVorticite(u):
# 	""" Autre calcul de vorticite """
# 	terme_x = diff(u, 2, 0)
# 	terme_y = diff(u, 1, 2)
# 	terme_z = diff(u, 0, 1)
# 	resultat = np.array([terme_y, terme_y, terme_z])
# 	return resultat

#calcul de la vorticite
def calculVorticite(ux, uy, uz):
	""" Autre calcul de vorticite, on est sur qu'il est bon """
	duzdy = np.roll(uz, -1, axis = 0) - np.roll(uz, 1, axis = 0)
	duydz = np.roll(uy, -1, axis = 2) - np.roll(uy, 1, axis = 2)
	terme_x = duzdy - duydz

	duxdz = np.roll(ux, -1, axis = 2) - np.roll(ux, 1, axis = 2)
	duzdx = np.roll(uz, -1, axis = 1) - np.roll(uz, 1, axis = 1)
	terme_y = duxdz - duzdx

	duydx = np.roll(uy, -1, axis = 1) - np.roll(uy, 1, axis = 1)
	duxdy = np.roll(ux, -1, axis = 0) - np.roll(ux, 1, axis = 0)
	terme_z = duydx - duxdy


	resultat = np.array([terme_y, terme_y, terme_z])
	return resultat

#Fonction de propagation dans toutes les directions
def propagation():
	global f, vitesses
	
	#Calcul de rho
	rho = np.sum(f, axis = 0)
	
	#Propagation
	for i in range(N):

		#on fait la propagation axe par axe
		dx, dy, dz = vitesses[i]
		f[i] = np.roll(f[i], dy, axis = 0)
		f[i] = np.roll(f[i], dx, axis = 1)
		f[i] = np.roll(f[i], dz, axis = 2)

	#on garde f en copie pour l'operation suivante
	f_copy = deepcopy(f)
	#collision avec les obstacles
	for i in range(1, N):
		#On recupere la direction "opposee"
		autre = autres[i]
		#On echange les directions opposees
		f[i][barrier[i]] = f_copy[autre][barrier[0]]


#Cette fonction gere le terme de collisions
def collisions():
	global f, u0, u02, omega

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

	#on evite de recalculer N fois ce terme constant
	terme1 = 1 - 1.5 * (ux ** 2 + uy ** 2 + uz ** 2) / nu2

	#on applique la formule BGK
	for i in range(N):
		ps = vitesses[i, 0] * ux + vitesses[i, 1] * uy + vitesses[i, 2] * uz
		terme2 = 3 * ps / nu
		terme3 = 4.5 * np.power(ps, 2) / nu2
		# f[i] = (1 - omega) * f[i] + omega * poids[i] * rho * (terme1 + terme2 + terme3)
		#on capture les erreurs d'overflow
		try:
			f[i] = (1 - omega) * f[i] + omega * poids[i] * rho * (terme1 + terme2 + terme3)
		except RuntimeWarning:
			print('Erreur d\'overflow')
	
	#on ajoute le champ de force
	for i in range(N):
		ps = np.dot(vitesses[i], F)
		f[i] += poids[i] * ps / nu
	
	#On force le flow en entree

	#Calcul du terme recurrent
	terme1 = 1 - 1.5 * u02 / nu2

	#on ne s'occupe que des directions concernees par le flow horizontal
	for i in [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
		ps = vitesses[i, 0] * u0
		terme2 = 3 * ps / nu
		terme3 = 4.5 * np.abs(vitesses[i, 0]) * u02 / nu2

		f[i, :, 0, :] = poids[i] * (terme1 + terme2 + terme3)


if __name__ == '__main__':
	######## On definit la barriere ###########
	barrier = np.zeros((N, largeur_cuve, longueur_cuve, epaisseur_cuve), dtype = bool)

	for x in range(longueur_cuve):
		for y in range(largeur_cuve):
			for z in range(epaisseur_cuve):
				if zoneBarriere(x, y, z):
					barrier[0, y, x, z] = True

	# barrier[0, :hauteur_barriere, :largeur_barriere, :] = True
	# barrier[0, largeur_cuve - hauteur_barriere:, :largeur_barriere, :] = True

	#Ces booleens servent a gerer les collisions
	for i in range(N):
		dx, dy, dz = vitesses[i]
		#On calcule les points aux limites des barrieres
		barrier[i] = np.roll(barrier[0], dy, axis = 0)
		barrier[i] = np.roll(barrier[i], dx, axis = 1)
		barrier[i] = np.roll(barrier[i], dz, axis = 2)

	########## On initialise avec un flux constant ################
	f = np.ones((N, largeur_cuve, longueur_cuve, epaisseur_cuve))
	terme1 = 1 - 1.5 * u02 / nu2
	
	for i in range(N):
		#produit scalaire de la vitesse forcee avec la vitesse horizontale
		ps = vitesses[i, 0] * u0
		terme2 = 3 * ps / nu
		terme3 = 4.5 * np.abs(vitesses[i, 0]) * u02 / nu2

		f[i] = poids[i] * (terme1 + terme2 + terme3)

	for step in range(30): #A ajuster pour que ça soit bien
		propagation()
		collisions()

	u0 = 0

	for step in range(30): #A ajuster pour que ça soit bien
		propagation()
		collisions()

	
	#Calcul des valeurs macroscopiques
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
	# fig, ax1 = plt.subplots(1, 1, projection = '3d')
	fig = p.figure()
	ax1 = p3.Axes3D(fig)
	ax1.set_xlabel('Y')
	ax1.set_ylabel('X')
	ax1.set_zlabel('Z')
	ax1.set_xlim(0, largeur_cuve)
	ax1.set_ylim(0, longueur_cuve)
	ax1.set_zlim(0, epaisseur_cuve)
	ax1.autoscale(enable = False)
	# x, y, z = np.meshgrid(
	# 	np.arange(0, longueur_cuve),
	# 	np.arange(0, largeur_cuve),
	# 	np.arange(0, epaisseur_cuve))

	w = calculVorticite(ux, uy, uz)
	w2 = w[0] ** 2 + w[1] ** 2 + w[2] ** 2
	m = np.max(w2)
	X = []
	Y = []
	Z = []
	wx = []
	wy = []
	wz = []

	for x in range(longueur_cuve):
		for y in range(largeur_cuve):
			for z in range(epaisseur_cuve):
				if w2[y, x, z] >= 0.3 * m:
					X.append(x)
					Y.append(y)
					Z.append(z)
					wx.append(w[0, y, x, z])
					wy.append(w[1, y, x, z])
					wz.append(w[2, y, x, z])


	fluidImage = ax1.quiver(Y, X, Z, wy, wx, wz, normalize = True)

	# Affichage vorticite
	# fluidImage = ax1.imshow(
	# 	w[2, :, :, zc], 
	# 	origin = 'lower', 
	# 	norm = plt.Normalize(-.1,.1), 
	# 	cmap = plt.get_cmap('jet'), 
	# 	interpolation = 'none'
	# )

	Xb = []
	Yb = []
	Zb = []
	for x in range(longueur_cuve):
		for y in range(largeur_cuve):
			for z in range(epaisseur_cuve):
				if barrier[0, y, x, z]:
					Xb.append(x)
					Yb.append(y)
					Zb.append(z)
	# barrierImage = ax1.scatter3D(Xb, Yb, Zb, c = 'black')
	#affichage des barrieres
	# bImageArray = np.zeros((largeur_cuve, longueur_cuve, 4), np.uint8)
	# bImageArray[barrier[0, :, :, zc], 3] = 255
	# barrierImage = ax1.imshow(bImageArray, origin = 'lower', interpolation = 'none')

	#fonction d'animation
	def nextFrame(k, *args):
		global u0, ux, uy, uz, barrier

		for step in range(30): #A ajuster pour que ça soit bien
			propagation()
			collisions()

		#Calcul des valeurs macroscopiques
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

		w = calculVorticite(ux, uy, uz)

		#on modifie l'image
		fluidImage.set_array(w[2, :, :, zc])

		#on annule la vitesse
		#C'est comme si l'obstacle s'arretait de bouger
		if k > 0:
			u0 = 0

		return (fluidImage, barrierImage)


	#animation
	# ani = animation.FuncAnimation(
	# 	fig, 
	# 	nextFrame, 
	# 	frames = range(100), 
	# 	repeat = False, 
	# 	interval = 1, 
	# 	blit = True)

	p.show()