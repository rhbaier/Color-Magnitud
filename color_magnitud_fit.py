 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:15:32 2023

@author: rbaier
"""

import astropy.table as tbd
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib.gridspec import GridSpec
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.stats import norm
from scipy.optimize import root_scalar, minimize
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

#funcion para estimar masa estelar de galaxia
#Taylor et al. 2011
def log_stellar_mass(color, absolute_magnitude_i):
    l_M=1.15 + 0.7*(color)-0.4*absolute_magnitude_i
    return(l_M)

cluster='Horologium'

fileloc_csv = '../tarea2/csv_data/csv_final/' #path donde está la tabla csv con los datos del cumulo

dist_gal_sk = '../tarea2/csv_data/'+cluster+'__Legacy_LS_SNMAD_4-Res5XR200_ALL_DistCals.csv'


dist_table = tbd.Table.read(dist_gal_sk)

dist_mpc=dist_table['distmpc']


#revisa todos los datos dentro del directorio que tienen la extension csv (en este ejemplo es solo uno, pero puede extenderse a N archivos)
file_extension_csv = '.csv'
files_in_directory_csv = [f for f in os.listdir(fileloc_csv) if f.endswith(file_extension_csv) if f!='clusters_holorogium_degrees.csv']
files_in_directory_csv.sort()

data_tab = tbd.Table.read(fileloc_csv+files_in_directory_csv[0]) #si ubieran más, se hace un ciclo for y agregamos el indice en vez de 0.

R_MAG=data_tab['RMAG'] #revisar bien cómo estan los nombres de cada columna en el archivo .csv a usar
G_MAG=data_tab['GMAG'] #pudiese ser r_mag, mag_r, etc...
I_MAG=data_tab['IMAG']
RA=data_tab['RA_J2000']
Dec=data_tab['Dec_J2000']
z_phot=data_tab['z_phot_mean'] #redshift fotometrico
color_GR=G_MAG-R_MAG #el corte en magnitud se hace por la falta de fuentes rojas tenues en los datos
color_GI=G_MAG-I_MAG
mag_cut= 18.5
color_cut_max=1.0
color_cut_min=0.1

mascara0=R_MAG<=mag_cut #corte en magnitud (para caso de Horologium anda bien este valor, en otro caso puede ser menor o mayor)

mascara0_1=R_MAG>=12.5 #por si se quiere quitar las galaxias mas brillantes, multiplicar por esta mascara (agregar el valor que se requiera)


mascara1=color_GR<=color_cut_max#corte en color (tambien puede variar tanto para mayor o menor valores)
mascara2=color_GR>=color_cut_min
mascara=mascara1*mascara2*mascara0*mascara0_1

#datos de magnitud y color g-r filtrados por los cortes en color y magnitud
RA=RA[mascara]
Dec=Dec[mascara]
R_MAG=R_MAG[mascara]
G_MAG=G_MAG[mascara]
I_MAG=I_MAG[mascara]
color_GR=color_GR[mascara]
dist_mpc=dist_mpc[mascara]
color_GI=color_GI[mascara] #este color se emplea para estimar la masa (Taylor et al. 2011)
z_phot=z_phot[mascara]

#%%
n_bins = 5  # Número de bins para calcular los ajustes gaussianos
length_bin = (max(R_MAG) - min(R_MAG)) / n_bins  # Define la longitud del bin basada en los valores de magnitud min y max
min_bin = min(R_MAG)
colors=['b','r','magenta','orange','green', 'cyan', 'pink', 'yellow', 'grey']


fig = plt.figure(figsize=(12, 8))
gs = GridSpec(1, 2, width_ratios=[7, 2])  # 2 columnas, la primera es más grande

# Subplot grande a la izquierda
ax1 = fig.add_subplot(gs[0])
ax1.scatter(R_MAG, color_GR, s=0.6, alpha=0.5)
ax1.invert_xaxis()
ax1.set_ylabel('g-r', fontsize=18)
ax1.set_xlabel('r_mag', fontsize=18)



# Subplots más pequeños a la derecha (uno encima del otro)
subplots_right = GridSpec(n_bins, 1, figure=fig, left=0.7, right=0.97, hspace=0.9, wspace=0.5)


#De aqui para abajo se hacen los ajustes gaussianos



color_mean_bins_rs=[] #estos son los dos arreglos mas importantes porque tienen los promedios determinados para la gaussiana mas roja (rs=red sequence)
color_sigma_bins_rs=[]
magnitud_median_bins=[] #y también la media del bin en magnitud usado para el ajuste

#arreglos con las intersecciones entre las dos gaussianas que se ajustan
color_inter_x=[]
mag_inter_y=[]
for i in range(n_bins):
    mask1 = R_MAG >= min_bin
    mask2 = R_MAG <= min_bin + length_bin
    mask = mask1 * mask2
    R_MAG_step = R_MAG[mask]
    color_GR_step = color_GR[mask]
    ax1.scatter(R_MAG_step, color_GR_step,color=colors[i], s=0.6, alpha=0.5)
    ax1.annotate(str(min(R_MAG))+'<R_MAG<' +str(mag_cut),(14,0.2), fontsize=8, bbox=dict(boxstyle="round,pad=0.4", fc="white"), zorder=4)
    ax1.annotate(str(color_cut_min)+'<G-R<' +str(color_cut_max),(14,0.16), fontsize=8, bbox=dict(boxstyle="round,pad=0.4", fc="white"), zorder=4)
    ax = fig.add_subplot(subplots_right[i])
    if i==0: #pra el caso particular del primer bin (recordar que esta ordenado de menor a mayor magnitud) se ajusta solo una gaussiana
        color_GR_step=np.ravel(color_GR_step).astype(float)
        ax.hist(color_GR_step, bins=10, color=colors[i],alpha=0.5, label='bin N°' + str(i + 1), density=True)
        mag_median=np.median(R_MAG_step) #usamos la mediana en el eje de la magnitud como el valor representativo de cada bin
        gmm_bin_n=GaussianMixture(n_components=1, random_state=0)
        gmm_bin_n.fit(color_GR_step.reshape(-1, 1))
        mean_bin_n=float(gmm_bin_n.means_[0])
        sigma_bin_n=np.sqrt(float(gmm_bin_n.covariances_[0][0]))
        A_n_1=gmm_bin_n.weights_[0]
        A_n_1_2=np.sqrt(1 / (2 * np.pi * sigma_bin_n**2))
        print('amplitud weight:',A_n_1)
        print('amplitud sigma based:',A_n_1_2)
        print('Mean gaussian fit:', mean_bin_n)
        print('Std gaussian fit:', sigma_bin_n)
        x=np.linspace(0.25, 1.5, 1000)
        y=A_n_1*stats.norm.pdf(x,mean_bin_n,sigma_bin_n).ravel()
        ax.plot(x,y, color='k', alpha=0.5)
        ax.axvline(x=mean_bin_n,color='k', alpha=0.4,linestyle='--')
        color_mean_bins_rs.append(mean_bin_n)
        color_sigma_bins_rs.append(sigma_bin_n)
        magnitud_median_bins.append(mag_median)
        ax1.scatter(mag_median,mean_bin_n, color='k',  edgecolors='k', marker='+', s=25, linewidths=13, alpha=0.3)
    else:
        ax.hist(color_GR_step, bins=50, color=colors[i],alpha=0.5, label='bin N°' + str(i + 1), density=True)
        mag_median=np.median(R_MAG_step)
        gmm_bin_n_2=GaussianMixture(n_components=2, init_params='k-means++', covariance_type='full', random_state=0)
        gmm_bin_n_2.fit(color_GR_step.reshape(-1, 1))
        mean_bin_n_2=float(gmm_bin_n_2.means_[0])
        sigma_bin_n_2=np.sqrt(float(gmm_bin_n_2.covariances_[0][0]))
        A_n_2=gmm_bin_n_2.weights_[0]
        mean_bin_n_2_2=float(gmm_bin_n_2.means_[1])
        sigma_n_bin_2_2=np.sqrt(float(gmm_bin_n_2.covariances_[1][0]))
        A_n_2_2=gmm_bin_n_2.weights_[1]
        print('mean 1:',mean_bin_n_2)
        print('mean 2:',mean_bin_n_2_2)
        promedios=[mean_bin_n_2, mean_bin_n_2_2]
        sigmas=[sigma_bin_n_2,sigma_n_bin_2_2]
        indice_maximo = np.argmax(promedios)
        def gaussian1(x):
            return A_n_2 * norm.pdf(x, mean_bin_n_2, sigma_bin_n_2)

        def gaussian2(x):
            return A_n_2_2 * norm.pdf(x, mean_bin_n_2_2, sigma_n_bin_2_2)
        def difference_function(x):
            return np.abs(gaussian1(x) - gaussian2(x))
        
        x2=np.linspace(mean_bin_n_2-0.8, mean_bin_n_2+0.8, 10000)
        y2=A_n_2*stats.norm.pdf(x2,mean_bin_n_2,sigma_bin_n_2).ravel()
        x2_2=np.linspace(mean_bin_n_2_2-0.8, mean_bin_n_2_2+0.8, 10000)
        y2_2=A_n_2_2*stats.norm.pdf(x2_2,mean_bin_n_2_2,sigma_n_bin_2_2).ravel()
        
        initial_guess = np.mean([mean_bin_n_2,mean_bin_n_2_2])  #suponemos un punto de interseccion que esta entre los dos peaks de las gaussianas
        intersection_result = minimize(difference_function, initial_guess, method='Nelder-Mead')
        
        # Punto de intersección
        interseccion_x = intersection_result.x[0]
        interseccion_y1 = gaussian1(interseccion_x)
        interseccion_y2 = gaussian2(interseccion_x)
        indice_interseccion = np.argmin(np.abs(y2 - y2_2))
        ax.plot(x2,y2, color='k', alpha=0.5)
        ax.plot(x2_2,y2_2, color='k', alpha=0.5)
        ax.axvline(x=mean_bin_n_2, color='k', alpha=0.4,linestyle='--')
        ax.axvline(x=mean_bin_n_2_2,color='k', alpha=0.4,linestyle='--')
        ax.scatter(interseccion_x, interseccion_y2, color='b', s=50, marker='o')
        magnitud_median_bins.append(mag_median)
        color_mean_bins_rs.append(promedios[indice_maximo]) #se guarda el centro de la gaussiana mas rojo (que es de la secuencia roja)
        color_sigma_bins_rs.append(sigmas[indice_maximo])
        color_inter_x.append(interseccion_x)
        #mag_inter_y.append(y2[interseccion_max])
        ax1.scatter(mag_median,max(promedios), color='k', marker='+', s=25, linewidths=13, alpha=0.7)
    ax.set_ylabel('Counts', fontsize=12, labelpad=0)
    if i==max(range(n_bins)):
        ax.set_xlabel('g-r', fontsize=12)
    ax.legend()
    ax.set_xlim(0,1.5)
    min_bin = min_bin + length_bin

for ax in fig.get_axes():
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color("black")
        ax.spines[axis].set_zorder(0)
    ax.tick_params(labelsize=12, which='both', direction='in', width=2.5, length=6, pad=0, color='k', colors='k')


#pasamos todo a arreglos numpy para evitar cualquier problema con operaciones

color_mean_bins_rs=np.array(color_mean_bins_rs) #estos son los dos arreglos mas importantes porque tienen los promedios determinados para la gaussiana mas roja (rs=red sequence)
magnitud_median_bins=np.array(magnitud_median_bins) #y también la media del bin en magnitud usado para el ajuste
color_sigma_bins_rs=np.array(color_sigma_bins_rs)
color_inter_x=np.array(color_inter_x)

#%% #ahora ajustamos un polinomio de grado 1 (una linea recta), a los valores de promedios obtenidos en la Red Sequence

orden=1 #orden del polinomio
coefficients_1 = np.polyfit(magnitud_median_bins, color_mean_bins_rs,orden) #se busca los parametros (m y b de la recta)

coefficients_2 = np.polyfit(magnitud_median_bins[1:], color_inter_x,orden) #se busca los parametros (m y b de la recta)



poly_function = np.poly1d(coefficients_1)

poly_function_inter = np.poly1d(coefficients_2)

color_mas_sigma_bin=color_mean_bins_rs+np.mean(color_sigma_bins_rs)

color_menos_sigma_bin=color_mean_bins_rs-np.mean(color_sigma_bins_rs)

mag_fit = np.linspace(min(R_MAG), max(R_MAG), 200) #red sequence ajustada

color_fit = poly_function(mag_fit)
color_inter_fit = poly_function_inter(mag_fit)



min_bin = min(R_MAG)

color_mas_sigma_bin=color_fit+np.mean(color_sigma_bins_rs)

color_menos_sigma_bin=color_fit-np.mean(color_sigma_bins_rs)


#de aqui para abajo es solo para mostrar el ajuste en el diagrama color-magnitud
colors=['b','r','magenta','orange','green', 'cyan', 'pink']


fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))


ax2.set_ylabel('g-r', fontsize=18)
ax2.set_xlabel('r_mag', fontsize=18)

ax2.set_xlim(min(R_MAG),max(R_MAG))

ax2.invert_xaxis()
for i in range(n_bins):
    mask1 = R_MAG >= min_bin
    mask2 = R_MAG <= min_bin + length_bin
    mask = mask1 * mask2
    R_MAG_step = R_MAG[mask]
    color_GR_step = color_GR[mask]
    ax2.scatter(R_MAG_step, color_GR_step,color=colors[i], s=0.6, alpha=0.8)
    min_bin = min_bin + length_bin
    

for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(1.5)
    ax2.spines[axis].set_color("black")
    ax2.spines[axis].set_zorder(0)
ax2.tick_params(labelsize=18, which='both', direction='in', width=2.5, length=6, pad=0, color='k', colors='k')

ax2.scatter(magnitud_median_bins,color_mean_bins_rs, color='k', marker='+', s=25, linewidths=13, alpha=0.7, label=r'$\mu_{0}$' + ' of the gaussian fit')

ax2.plot(mag_fit,color_fit, color='k', alpha=0.9, label='Best red-sequence linear fit')
ax2.plot(mag_fit,color_inter_fit, color='blue', alpha=0.9, linestyle='dashdot',label='Best gaussian intersection linear fit')

ax2.scatter(magnitud_median_bins[1:],color_inter_x, linewidth=2, color='blue', s=50, label='Gaussian intersections')

#ax2.plot(magnitud_median_bins[1:],color_inter_x, linewidth=2, color='blue', linestyle='-.', label='Intersection gaussian fits')
ax2.plot(mag_fit,color_mas_sigma_bin, color='k', linestyle='--', alpha=0.6, label=r'$1 \sigma$' + ' (mean) 1-D gaussian fit')
ax2.plot(mag_fit,color_menos_sigma_bin, color='k',linestyle='--', alpha=0.6)
ax2.legend(loc='lower right')

#%%Se va considerar la separacion de blue y red como el ajuste de las intersecciones de las gaussianas

estimacion=np.interp(R_MAG,mag_fit,color_inter_fit)

color_red_galaxies, mag_red_galaxies= color_GR[color_GR>estimacion], R_MAG[color_GR>estimacion]

color_blue_galaxies, mag_blue_galaxies= color_GR[color_GR<estimacion], R_MAG[color_GR<estimacion]

RA_red=RA[color_GR>estimacion]
Dec_red=Dec[color_GR>estimacion]

RA_blue=RA[color_GR<estimacion]
Dec_blue=Dec[color_GR<estimacion]



cosmo=FlatLambdaCDM(70, 0.3)


size_Horologium_line_of_see=cosmo.luminosity_distance(np.mean(z_phot))-cosmo.luminosity_distance(min(z_phot))

z_cl = 0.06415

dist_galaxies=cosmo.luminosity_distance(np.ones(len(z_phot))*z_cl)
dist_galaxies=np.array(dist_galaxies)*1e+6 #distancia en parsecs a (a la Tierra!!!!!!!!)

Absolute_i_mag= I_MAG-5*np.log10(dist_galaxies/10)

L_stellar_mass=log_stellar_mass(color_GI, Absolute_i_mag)
stellar_mass_cut=11#np.mean(L_stellar_mass)+1


L_stellar_mass_red=L_stellar_mass[color_GR>estimacion]
L_stellar_mass_blue=L_stellar_mass[color_GR<estimacion]

stellar_mass_cut_blue=10#np.mean(L_stellar_mass_blue)
stellar_mass_cut_red=10#np.mean(L_stellar_mass_red)
mask_stellar_mass=L_stellar_mass>stellar_mass_cut

mask_stellar_mass_blue=L_stellar_mass_blue>stellar_mass_cut_blue#np.mean(L_stellar_mass_blue)
mask_stellar_mass_red=L_stellar_mass_red>stellar_mass_cut_red#np.mean(L_stellar_mass_red)






#%%

fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))


ax2.set_ylabel('g-r', fontsize=18)
ax2.set_xlabel('r_mag', fontsize=18)

ax2.set_xlim(min(R_MAG),max(R_MAG))

ax2.invert_xaxis()
ax2.scatter(mag_red_galaxies, color_red_galaxies,color='red', s=0.6, alpha=0.8)
ax2.scatter(mag_blue_galaxies, color_blue_galaxies,color='blue', s=0.6, alpha=0.8)


for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(1.5)
    ax2.spines[axis].set_color("black")
    ax2.spines[axis].set_zorder(0)
ax2.tick_params(labelsize=18, which='both', direction='in', width=2.5, length=6, pad=0, color='k', colors='k')

#ax2.scatter(magnitud_median_bins,color_mean_bins_rs, color='k', marker='+', s=25, linewidths=13, alpha=0.7, label=r'$\mu_{0}$' + ' of the gaussian fit')

ax2.plot(mag_fit,color_fit, color='k', alpha=0.9, label='Best red-sequence linear fit')
ax2.plot(mag_fit,color_inter_fit, color='blue', alpha=0.9, linestyle='dashdot',label='Best gaussian intersection linear fit')

#ax2.scatter(magnitud_median_bins[1:],color_inter_x, linewidth=2, color='blue', s=50, label='Gaussian intersections')

#ax2.plot(magnitud_median_bins[1:],color_inter_x, linewidth=2, color='blue', linestyle='-.', label='Intersection gaussian fits')
ax2.plot(mag_fit,color_mas_sigma_bin, color='k', linestyle='--', alpha=0.6, label=r'$1 \sigma$' + ' (mean) 1-D gaussian fit')
ax2.plot(mag_fit,color_menos_sigma_bin, color='k',linestyle='--', alpha=0.6)
ax2.legend(loc='lower right')

#%%

fileloc_segs = '../tarea2/'

file_extension_segs = '.segs'
files_in_directory_segs = [f for f in os.listdir(fileloc_segs) if f.endswith(file_extension_segs)]
files_in_directory_segs.sort()

clusters_holorogium=tbd.Table.read('/home/dell-inspiron-15/Documents/topico_galaxias/tarea2/csv_data/'+'clusters_holorogium_degrees.csv')

ra_clusters_holorogium=clusters_holorogium['RA_J2000']
dec_clusters_holorogium=clusters_holorogium['Dec_J2000']
names_clusters=clusters_holorogium['name']

U0,U1,V0,V1=np.loadtxt(fileloc_segs+files_in_directory_segs[0], usecols=(0,1,2,3), comments='#',unpack=True)
U02,U12,V02,V12=np.loadtxt(fileloc_segs+files_in_directory_segs[1], usecols=(0,1,2,3), comments='#',unpack=True)

mask1 = ra_clusters_holorogium>min(RA)#mask para cortar en subestructuras dentro de los datos que se tienen para chances
mask2 = ra_clusters_holorogium<max(RA)
mask3 = dec_clusters_holorogium>min(Dec)
mask4 = dec_clusters_holorogium<max(Dec)
ra_clusters_holorogium = ra_clusters_holorogium[mask1*mask2*mask3*mask4]
dec_clusters_holorogium = dec_clusters_holorogium[mask1*mask2*mask3*mask4]
names_clusters=names_clusters[mask1*mask2*mask3*mask4]

#%%
fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
#for g,t,k,l in zip(U0,V0,U1,V1):  #plot skeleton
#    ax1.plot([g,t],[k,l],'r', alpha=1,zorder=2, linewidth=2.2)


for g,t,k,l in zip(U02,V02,U12,V12):  #plot skeleton
    ax1.plot([g,t],[k,l],'blue', alpha=1,zorder=2, linewidth=2.2)

cmap = plt.get_cmap('gray_r')
#hist, xedges, yedges = np.histogram2d(ra, dec, bins=(150, 150))
#hist = hist.T  # Transpone el histograma


plt.hexbin(RA, Dec, gridsize=50, cmap='gray_r')
ax1.scatter(ra_clusters_holorogium,dec_clusters_holorogium, zorder=3, color='cyan', edgecolor='b', label='Abell et al. 1989; Struble et al. 1999')  


for j in range(len(ra_clusters_holorogium)):
    x = ra_clusters_holorogium[j]
    y = dec_clusters_holorogium[j]
    name = names_clusters[j]

    # Dibuja una caja alrededor del texto
    ax1.annotate(name, (x+0.3, y+0.4), fontsize=8, bbox=dict(boxstyle="round,pad=0.1", fc="white"), zorder=4)



ax1.invert_xaxis()
ax1.grid(False)
ax1.legend(loc='upper left')
ax1.set_xlabel('RA', fontsize=18)
ax1.set_ylabel('Dec', fontsize=18)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.5)
    ax1.spines[axis].set_color("black")
    ax1.spines[axis].set_zorder(0)
ax1.tick_params(labelsize=18, which='both', direction='in', width=2.5, length=6, pad=0, color='k', colors='k')
ax1.set_aspect('equal')



#%%
fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
#for g,t,k,l in zip(U0,V0,U1,V1):  #plot skeleton
#    ax1.plot([g,t],[k,l],'r', alpha=1,zorder=2, linewidth=2.2)


for g,t,k,l in zip(U02,V02,U12,V12):  #plot skeleton
    ax1.plot([g,t],[k,l],'blue', alpha=1,zorder=2, linewidth=2.2)

cmap_mass = plt.get_cmap('inferno_r')
#hist, xedges, yedges = np.histogram2d(ra, dec, bins=(150, 150))
#hist = hist.T  # Transpone el histograma

scatter_plot = ax1.scatter(RA[mask_stellar_mass], Dec[mask_stellar_mass], c=L_stellar_mass[mask_stellar_mass], cmap=cmap_mass, s=10,vmin=10.8, vmax=np.max(L_stellar_mass), alpha=0.8)

# Añadir una barra de color
cbar = plt.colorbar(scatter_plot)
cbar.set_label(r'$\log(M_{*}/M_{\odot})$',fontsize=14)

# Ajustar zorder para la barra de color
cbar.solids.set_edgecolor("face")
cbar.set_alpha(1)
# Configurar un valor de zorder menor para asegurar que la barra de color esté detrás de los puntos
cbar.ax.set_zorder(1)
ax1.set_zorder(0)
ax1.scatter(ra_clusters_holorogium,dec_clusters_holorogium, zorder=3, color='cyan', edgecolor='b')  
ax1.annotate(r'$\log(M_{*}/M_{\odot})>$'+str(round(stellar_mass_cut,1)), (57, -43), fontsize=13, bbox=dict(boxstyle="round,pad=0.3", fc="white"), zorder=4)


for j in range(len(ra_clusters_holorogium)):
    x = ra_clusters_holorogium[j]
    y = dec_clusters_holorogium[j]
    name = names_clusters[j]

    # Dibuja una caja alrededor del texto
    ax1.annotate(name, (x+0.3, y+0.4), fontsize=8, bbox=dict(boxstyle="round,pad=0.1", fc="white"), zorder=4)


ax1.invert_xaxis()
ax1.grid(False)
#ax1.legend()
ax1.set_xlabel('RA', fontsize=18)
ax1.set_ylabel('Dec', fontsize=18)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.5)
    ax1.spines[axis].set_color("black")
    ax1.spines[axis].set_zorder(0)
ax1.tick_params(labelsize=18, which='both', direction='in', width=2.5, length=6, pad=0, color='k', colors='k')
ax1.set_aspect('equal')

#%% filament view red galaxies

fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
#for g,t,k,l in zip(U0,V0,U1,V1):  #plot skeleton
#    ax1.plot([g,t],[k,l],'r', alpha=1,zorder=2, linewidth=2.2)


for g,t,k,l in zip(U02,V02,U12,V12):  #plot skeleton
    ax1.plot([g,t],[k,l],'blue', alpha=1,zorder=2, linewidth=2.2)

cmap_mass = plt.get_cmap('Reds')
#hist, xedges, yedges = np.histogram2d(ra, dec, bins=(150, 150))
#hist = hist.T  # Transpone el histograma

scatter_plot = ax1.scatter(RA_red[mask_stellar_mass_red], Dec_red[mask_stellar_mass_red], c=L_stellar_mass_red[mask_stellar_mass_red], cmap=cmap_mass, s=7,vmin=9.5, vmax=np.max(L_stellar_mass_red[mask_stellar_mass_red]), alpha=0.8)

# Añadir una barra de color
cbar = plt.colorbar(scatter_plot)
cbar.set_label(r'$\log(M_{*}/M_{\odot})$',fontsize=14)

# Ajustar zorder para la barra de color
cbar.solids.set_edgecolor("face")
cbar.set_alpha(1)
# Configurar un valor de zorder menor para asegurar que la barra de color esté detrás de los puntos
cbar.ax.set_zorder(1)
ax1.set_zorder(0)
ax1.scatter(ra_clusters_holorogium,dec_clusters_holorogium, zorder=3, color='cyan', edgecolor='b')  
ax1.annotate(r'$\log(M_{*}/M_{\odot})>$'+str(round(stellar_mass_cut_red,1)), (57, -43), fontsize=13, bbox=dict(boxstyle="round,pad=0.3", fc="white"), zorder=4)


for j in range(len(ra_clusters_holorogium)):
    x = ra_clusters_holorogium[j]
    y = dec_clusters_holorogium[j]
    name = names_clusters[j]

    # Dibuja una caja alrededor del texto
    ax1.annotate(name, (x+0.3, y+0.4), fontsize=8, bbox=dict(boxstyle="round,pad=0.1", fc="white"), zorder=4)


ax1.invert_xaxis()
ax1.grid(False)
#ax1.legend()
ax1.set_xlabel('RA', fontsize=18)
ax1.set_ylabel('Dec', fontsize=18)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.5)
    ax1.spines[axis].set_color("black")
    ax1.spines[axis].set_zorder(0)
ax1.tick_params(labelsize=18, which='both', direction='in', width=2.5, length=6, pad=0, color='k', colors='k')
ax1.set_aspect('equal')

#%% filament view blue galaxies

fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
#for g,t,k,l in zip(U0,V0,U1,V1):  #plot skeleton
#    ax1.plot([g,t],[k,l],'r', alpha=1,zorder=2, linewidth=2.2)


for g,t,k,l in zip(U02,V02,U12,V12):  #plot skeleton
    ax1.plot([g,t],[k,l],'blue', alpha=1,zorder=2, linewidth=2.2)

cmap_mass = plt.get_cmap('Blues')
#hist, xedges, yedges = np.histogram2d(ra, dec, bins=(150, 150))
#hist = hist.T  # Transpone el histograma

scatter_plot = ax1.scatter(RA_blue[mask_stellar_mass_blue], Dec_blue[mask_stellar_mass_blue], c=L_stellar_mass_blue[mask_stellar_mass_blue], cmap=cmap_mass, s=7,vmin=8, vmax=np.max(L_stellar_mass_blue[mask_stellar_mass_blue]), alpha=0.8)

# Añadir una barra de color
cbar = plt.colorbar(scatter_plot)
cbar.set_label(r'$\log(M_{*}/M_{\odot})$',fontsize=14)

# Ajustar zorder para la barra de color
cbar.solids.set_edgecolor("face")
cbar.set_alpha(1)
# Configurar un valor de zorder menor para asegurar que la barra de color esté detrás de los puntos
cbar.ax.set_zorder(1)
ax1.set_zorder(0)
ax1.scatter(ra_clusters_holorogium,dec_clusters_holorogium, zorder=3, color='cyan', edgecolor='b')  
ax1.annotate(r'$\log(M_{*}/M_{\odot})>$'+str(round(stellar_mass_cut_blue,1)), (57, -43), fontsize=13, bbox=dict(boxstyle="round,pad=0.3", fc="white"), zorder=4)


for j in range(len(ra_clusters_holorogium)):
    x = ra_clusters_holorogium[j]
    y = dec_clusters_holorogium[j]
    name = names_clusters[j]

    # Dibuja una caja alrededor del texto
    ax1.annotate(name, (x+0.3, y+0.4), fontsize=8, bbox=dict(boxstyle="round,pad=0.1", fc="white"), zorder=4)



ax1.invert_xaxis()
ax1.grid(False)
#ax1.legend()
ax1.set_xlabel('RA', fontsize=18)
ax1.set_ylabel('Dec', fontsize=18)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.5)
    ax1.spines[axis].set_color("black")
    ax1.spines[axis].set_zorder(0)
ax1.tick_params(labelsize=18, which='both', direction='in', width=2.5, length=6, pad=0, color='k', colors='k')
ax1.set_aspect('equal')

#%% filament view all galaxies

fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
#for g,t,k,l in zip(U0,V0,U1,V1):  #plot skeleton
#    ax1.plot([g,t],[k,l],'r', alpha=1,zorder=2, linewidth=2.2)



cmap_mass = plt.get_cmap('Reds')
cmap_mass_blue = plt.get_cmap('Blues')


#hist, xedges, yedges = np.histogram2d(ra, dec, bins=(150, 150))
#hist = hist.T  # Transpone el histograma

scatter_plot_red = ax1.scatter(RA_red, Dec_red, c=L_stellar_mass[color_GR>estimacion], cmap=cmap_mass, s=7,vmin=7, vmax=max(L_stellar_mass_red), alpha=0.8, zorder=3)

# Añadir una barra de color
cbar1 = plt.colorbar(scatter_plot_red)
cbar1.set_label(r'$\log(M_{*}/M_{\odot})$',fontsize=14)

scatter_plot_blue = ax1.scatter(RA_blue, Dec_blue, c=L_stellar_mass[color_GR<estimacion], cmap=cmap_mass_blue, s=7,vmin=7, vmax=max(L_stellar_mass_blue), alpha=0.8)

# Añadir una barra de color
cbar2 = plt.colorbar(scatter_plot_blue,location='left', pad=0.15)
cbar2.set_label(r'$\log(M_{*}/M_{\odot})$',fontsize=14)

# Ajustar zorder para la barra de color
cbar.solids.set_edgecolor("face")
cbar.set_alpha(1)
# Configurar un valor de zorder menor para asegurar que la barra de color esté detrás de los puntos
cbar.ax.set_zorder(1)
ax1.set_zorder(0)
ax1.scatter(ra_clusters_holorogium,dec_clusters_holorogium, zorder=3, color='yellow', edgecolor='b')  
#ax1.annotate(r'$\log(M_{*}/M_{\odot})>$'+str(round(np.mean(L_stellar_mass)+0.47,1)), (57, -43), fontsize=13, bbox=dict(boxstyle="round,pad=0.3", fc="white"), zorder=2)


for j in range(len(ra_clusters_holorogium)):
    x = ra_clusters_holorogium[j]
    y = dec_clusters_holorogium[j]
    name = names_clusters[j]

    # Dibuja una caja alrededor del texto
    ax1.annotate(name, (x+0.3, y+0.4), fontsize=8, bbox=dict(boxstyle="round,pad=0.1", fc="white"), zorder=8)
    
for g,t,k,l in zip(U02,V02,U12,V12):  #plot skeleton
    ax1.plot([g,t],[k,l],'k', alpha=1,zorder=6, linewidth=2.2)

ax1.invert_xaxis()
ax1.grid(False)
#ax1.legend()
ax1.set_xlabel('RA', fontsize=18)
ax1.set_ylabel('Dec', fontsize=18)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.5)
    ax1.spines[axis].set_color("black")
    ax1.spines[axis].set_zorder(0)
ax1.tick_params(labelsize=18, which='both', direction='in', width=2.5, length=6, pad=0, color='k', colors='k')
ax1.set_aspect('equal')

#%%mass_distance plot
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))


n_bins=4
length_bin = (max(dist_mpc) - min(dist_mpc))/n_bins  # Define la longitud del bin basada en los valores de distancia min y max

min_bin=min(dist_mpc)

x=[]
y=[]
y_red=[]
y_blue=[]

ax2.set_xlabel('filament distance (projected) [Mpc]', fontsize=18)
ax2.set_ylabel(r'$\log(M_{*}/M_{\odot})$', fontsize=18)

ax2.set_xlim(min(dist_mpc),max(dist_mpc))
ax2.set_ylim(10,max(L_stellar_mass))



#ax2.invert_xaxis()
for i in range(n_bins):
    mask1 = dist_mpc >= min_bin
    mask2 = dist_mpc <= min_bin + length_bin
    mask = mask1 * mask2
    dist_fil_step = dist_mpc[mask]
    mask_color_red= color_GR>estimacion 
    mask_color_blue= color_GR<estimacion 
    mask_mass=L_stellar_mass>10.5
    
    L_stellar_mass_step = L_stellar_mass[mask*mask_mass]
    L_stellar_mass_step_blue = L_stellar_mass[mask*mask_color_blue*mask_mass]
    L_stellar_mass_step_red=L_stellar_mass[mask*mask_color_red*mask_mass]

    dist_median_step=np.median(dist_fil_step)
    x.append(dist_median_step)


    L_mean_stellar_mass_step=np.mean(L_stellar_mass_step)
    L_mean_stellar_mass_step_blue=np.mean(L_stellar_mass_step_blue)
    L_mean_stellar_mass_step_red=np.mean(L_stellar_mass_step_red)
    y.append(L_mean_stellar_mass_step)
    y_blue.append(L_mean_stellar_mass_step_blue)
    y_red.append(L_mean_stellar_mass_step_red)
    L_error_stellar_mass_step=np.std(L_stellar_mass_step)
    L_error_stellar_mass_step_blue=np.std(L_stellar_mass_step_blue)
    L_error_stellar_mass_step_red=np.std(L_stellar_mass_step_red)
    import scipy.stats.distributions as dist
    c = 0.99 #intervalo de confianza
    k = len(y_blue) #número de éxitos
    n = len(y_blue) + len(RA) # número total de la muestra
    err_low = dist.beta.ppf((1-c)/2.,k+1,n-k+1) #estimación del valor menor del error
    err_up =  dist.beta.ppf(1-(1-c)/2.,k+1,n-k+1)# estimación del valor superior del error
    
    
    ax2.errorbar(dist_median_step, L_mean_stellar_mass_step, yerr=L_error_stellar_mass_step, fmt='none', ecolor='green', capsize=5)
    ax2.errorbar(dist_median_step,L_mean_stellar_mass_step_blue , yerr=L_error_stellar_mass_step_blue, fmt='none', ecolor='blue', capsize=5)
    ax2.errorbar(dist_median_step, L_mean_stellar_mass_step_red , yerr=L_error_stellar_mass_step_red, fmt='none', ecolor='red', capsize=5)
    #ax2.errorbar([dist_median_step],[len(y_blue)/(len(y_blue)+len(RA))],yerr=[(err_up-err_low)/2], marker='s',c='blue') # línea de código para plotear cuadrados en un punto de mi muestra

    #ax2.scatter(dist_median_step, L_mean_stellar_mass_step,color='green', s=50, alpha=0.8)
    min_bin = min_bin + length_bin
ax2.plot(x,y,color='green', markersize=8, marker='o', label='All galaxies') 
ax2.plot(x,y_blue,color='blue', markersize=8, marker='o', label='Blue cloud') 
ax2.plot(x,y_red,color='red', markersize=8, marker='o', label='Red sequence') 

ax2.legend()
for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(1.5)
    ax2.spines[axis].set_color("black")
    ax2.spines[axis].set_zorder(0)
ax2.tick_params(labelsize=18, which='both', direction='in', width=2.5, length=6, pad=0, color='k', colors='k')


#%%galaxy fractions plot blue-red galaxies
import statsmodels.api as sm
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))


n_bins=4
length_bin = (max(dist_mpc) - min(dist_mpc))/n_bins  # Define la longitud del bin basada en los valores de distancia min y max

min_bin=min(dist_mpc)



ax2.set_xlabel('filament distance (projected) [Mpc]', fontsize=18)
ax2.set_ylabel(r'$N_{gal}$' + ' (total sample, ' + str(len(RA)) + ' galaxies)', fontsize=18)

ax2.set_xlim(min(dist_mpc),max(dist_mpc))
#ax2.set_ylim(-0.5,1)



x_red=[]
y_red=[]

x_blue=[]
y_blue=[]

#ax2.invert_xaxis()
for i in range(n_bins):
    mask1 = dist_mpc >= min_bin
    mask2 = dist_mpc <= min_bin + length_bin
    mask_color_red= color_GR>estimacion 
    mask_color_blue= color_GR<estimacion 
    mask_red = mask1 * mask2 * mask_color_red
    mask_blue = mask1 * mask2 * mask_color_blue
    
    dist_fil_step_red = dist_mpc[mask1*mask2]
    L_stellar_mass_step_red = L_stellar_mass[mask_red]
    L_stellar_mass_step_blue = L_stellar_mass[mask_blue]
    dist_median_step_red=np.median(dist_fil_step_red)
    x_red.append(dist_median_step_red)
    x_blue.append(dist_median_step_red)
    y_blue.append(len(L_stellar_mass_step_blue))
    y_red.append(len(L_stellar_mass_step_red))
    import scipy.stats.distributions as dist
    c = 0.99 #intervalo de confianza
    k = len(x_blue) #número de éxitos
    n = len(x_blue) + len(RA) # número total de la muestra
    err_low = dist.beta.ppf((1-c)/2.,k+1,n-k+1) #estimación del valor menor del error
    err_up =  dist.beta.ppf(1-(1-c)/2.,k+1,n-k+1)# estimación del valor superior del error
    confianza_intervalo_blue = sm.stats.proportion_confint(len(L_stellar_mass_step_blue),len(RA), alpha=0.01)
    confianza_intervalo_red = sm.stats.proportion_confint(len(L_stellar_mass_step_blue),len(RA), alpha=0.01)
    ax2.errorbar(dist_median_step_red, len(L_stellar_mass_step_blue), yerr=np.mean(confianza_intervalo_blue), fmt='none', ecolor='blue', capsize=5)
    ax2.errorbar(dist_median_step_red, len(L_stellar_mass_step_red), yerr=np.mean(confianza_intervalo_red), fmt='none', ecolor='red', capsize=5)
    ax2.errorbar([dist_median_step],[len(x_blue)/(len(x_blue)+len(RA))],yerr=[(err_up-err_low)/2], marker='s',c='blue') # línea de código para plotear cuadrados en un punto de mi muestra

    min_bin = min_bin + length_bin
    
ax2.plot(x_red, y_red,color='red', marker='o', label='Red sequence')
ax2.plot(x_blue, y_blue,color='blue', marker='o', label='Blue cloud')
ax2.annotate('Confidence='+str(99) + '%', (6, 0.8), fontsize=13, bbox=dict(boxstyle="round,pad=0.3", fc="white"), zorder=2)

ax2.legend()


for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(1.5)
    ax2.spines[axis].set_color("black")
    ax2.spines[axis].set_zorder(0)
ax2.tick_params(labelsize=18, which='both', direction='in', width=2.5, length=6, pad=0, color='k', colors='k')

#%%galaxy fractions plot blue galaxies
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))


n_bins=4
length_bin = (max(dist_mpc) - min(dist_mpc))/n_bins  # Define la longitud del bin basada en los valores de distancia min y max

min_bin=min(dist_mpc)



ax2.set_xlabel('filament distance (projected) [Mpc]', fontsize=18)
ax2.set_ylabel(r'$f_{gal}$ ' + '(/ N° of gal. at a given distance)', fontsize=18)

ax2.set_xlim(min(dist_mpc),max(dist_mpc))
ax2.set_ylim(-0.5,1.5)



x_red=[]
y_red=[]

x_blue=[]
y_blue=[]

#ax2.invert_xaxis()
for i in range(n_bins):
    mask1 = dist_mpc >= min_bin
    mask2 = dist_mpc <= min_bin + length_bin
    mask_color_red= color_GR>estimacion 
    mask_color_blue= color_GR<estimacion 
    mask_mass=L_stellar_mass>1
    mask_red = mask1 * mask2 * mask_color_red
    mask_blue = mask1 * mask2 * mask_color_blue
    
    dist_fil_step_red = dist_mpc[mask1*mask2]
    L_stellar_mass_step_red = L_stellar_mass[mask_red]
    L_stellar_mass_step_blue = L_stellar_mass[mask_blue]
    dist_median_step_red=np.median(dist_fil_step_red)
    c = 0.99 #intervalo de confianza
    k = len(L_stellar_mass_blue) #número de éxitos
    n = len(RA) # número total de la muestra
    err_low = dist.beta.ppf((1-c)/2.,k+1,n-k+1) #estimación del valor menor del error
    err_up =  dist.beta.ppf(1-(1-c)/2.,k+1,n-k+1)# estimación del valor superior del error
    x_red.append(dist_median_step_red)
    x_blue.append(dist_median_step_red)
    y_blue.append(len(L_stellar_mass_step_blue)/len(RA[mask1*mask2]))
    y_red.append(len(L_stellar_mass_step_red)/len(RA[mask1*mask2]))
    
    confianza_intervalo_blue = sm.stats.proportion_confint(len(L_stellar_mass_step_blue),len(RA[mask1*mask2]), alpha=0.33)
    confianza_intervalo_red = sm.stats.proportion_confint(len(L_stellar_mass_step_blue),len(RA[mask1*mask2]), alpha=0.33)
    ax2.errorbar(dist_median_step_red, len(L_stellar_mass_step_blue)/len(RA[mask1*mask2]), yerr=np.mean(confianza_intervalo_blue)/2, fmt='none', ecolor='blue', capsize=5)
    ax2.errorbar(dist_median_step_red, len(L_stellar_mass_step_red)/len(RA[mask1*mask2]), yerr=np.mean(confianza_intervalo_red)/2, fmt='none', ecolor='red', capsize=5)
    #ax2.errorbar([dist_median_step],len(L_stellar_mass_step_blue)/len(RA[mask1*mask2]),yerr=[(err_up-err_low)/2], marker='s',c='blue') # línea de código para plotear cuadrados en un punto de mi muestra

    min_bin = min_bin + length_bin
    
ax2.plot(x_red, y_red,color='red', marker='o', label='Red sequence')
ax2.plot(x_blue, y_blue,color='blue', marker='o', label='Blue cloud')
ax2.annotate('Confidence='+str(66) + '%', (6, 0.8), fontsize=13, bbox=dict(boxstyle="round,pad=0.3", fc="white"), zorder=2)

ax2.legend()


for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(1.5)
    ax2.spines[axis].set_color("black")
    ax2.spines[axis].set_zorder(0)
ax2.tick_params(labelsize=18, which='both', direction='in', width=2.5, length=6, pad=0, color='k', colors='k')

#%%redshift histogram

np.random.seed(123)

random_z_phot = np.random.normal(loc=0.04, scale=0.002, size=300)

redshift_rojas=z_phot[color_GR>estimacion]

z_phot_modified=np.concatenate((z_phot,random_z_phot))


fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))


n_bins=4
length_bin = (max(dist_mpc) - min(dist_mpc))/n_bins  # Define la longitud del bin basada en los valores de distancia min y max

min_bin=min(dist_mpc)



ax2.set_xlabel('Photometric redshift', fontsize=18)
ax2.set_ylabel('Counts', fontsize=18)

#ax2.set_xlim(min(dist_mpc),max(dist_mpc))
#ax2.set_ylim(-0.5,1.5)


ax2.hist(z_phot_modified, bins=200,color='pink')



for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(1.5)
    ax2.spines[axis].set_color("black")
    ax2.spines[axis].set_zorder(0)
ax2.tick_params(labelsize=18, which='both', direction='in', width=2.5, length=6, pad=0, color='k', colors='k')



#%%Jackknife method


redshift = z_phot # o z_phot 


def calcular_promedio(redshift):
    return np.mean(redshift)

#implementamos el metodo de Jackknife
def jackknife(muestra, funcion_estimador):
    n = len(muestra)
    estimadores = []

    for i in range(n):
        muestra_jackknife = np.delete(muestra, i)
        estimador_jackknife = funcion_estimador(muestra_jackknife)
        estimadores.append(estimador_jackknife)

    return np.array(estimadores)

#se determinan los estimadores y el error
estimadores_jackknife = jackknife(redshift, calcular_promedio)

valor_promedio_original = calcular_promedio(redshift)
correccion = (len(redshift) - 1) * (valor_promedio_original - (1/len(redshift)) * np.sum(estimadores_jackknife))
valor_promedio_jackknife = valor_promedio_original + correccion
error_estandar = np.sqrt(np.sum((len(redshift) * valor_promedio_original - 
                                 valor_promedio_jackknife - (len(redshift) - 1) * estimadores_jackknife) ** 2) / 
                         (len(redshift) * (len(redshift) - 1)))






#%%redshift histogram mean value

np.random.seed(123)

random_z_phot = np.random.normal(loc=0.04, scale=0.002, size=300)

redshift_rojas=z_phot[color_GR>estimacion]

z_phot_modified=np.concatenate((z_phot,random_z_phot))


fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))


n_bins=4
length_bin = (max(dist_mpc) - min(dist_mpc))/n_bins  # Define la longitud del bin basada en los valores de distancia min y max

min_bin=min(dist_mpc)



ax2.set_xlabel('Photometric redshift', fontsize=18)
ax2.set_ylabel('Counts', fontsize=18)

#ax2.set_xlim(min(dist_mpc),max(dist_mpc))
#ax2.set_ylim(-0.5,1.5)


ax2.hist(z_phot_modified, bins=200,color='pink')
ax2.axvline(x=valor_promedio_jackknife, color='k', alpha=0.7,linestyle='-')
ax2.axvline(x=valor_promedio_jackknife+error_estandar, color='k', alpha=0.4,linestyle='--')
ax2.axvline(x=valor_promedio_jackknife-error_estandar, color='k', alpha=0.4,linestyle='--')




for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(1.5)
    ax2.spines[axis].set_color("black")
    ax2.spines[axis].set_zorder(0)
ax2.tick_params(labelsize=18, which='both', direction='in', width=2.5, length=6, pad=0, color='k', colors='k')


#%%distance histogram


print(f"Valor promedio original: {valor_promedio_original}")
print(f"Corrección (sesgo): {correccion}")
print(f"Valor promedio estimado (jackknife): {valor_promedio_jackknife}")
print(f"Error estándar estimado: {error_estandar}")




fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))


ax2.set_xlabel('Distance to nearest filament [Mpc]', fontsize=18)
ax2.set_ylabel('Counts', fontsize=18)

#ax2.set_xlim(min(dist_mpc),max(dist_mpc))
ax2.set_xlim(0,10)


ax2.hist(dist_mpc[color_GR>estimacion], bins=150,color='red', alpha=0.3, label='Red galaxies')
ax2.hist(dist_mpc[color_GR<estimacion], bins=150,color='blue', alpha=0.3, label='Blue galaxies')


ax2.legend()
for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(1.5)
    ax2.spines[axis].set_color("black")
    ax2.spines[axis].set_zorder(0)
ax2.tick_params(labelsize=18, which='both', direction='in', width=2.5, length=6, pad=0, color='k', colors='k')


