# Color-Magnitud
The idea of these two Python scripts is study the distribution of galaxies in a galaxy cluster 
as a function of the distance respect to the nearest filament, based on the followig simple steps:

1) First, using color_magnitud_fit.py, classify the galaxies based on the color-magnitud diagram.
This code identify the Red-Sequence and separates to the Blue-Cloud.

2) The second step is use DisPerSE (see [DisPerSE-running](https://github.com/rhbaier/DisPerSE-running.git) for more details
in filament detections using DisPerSE) to detect the filmentary structure on the data.

3) With the distcal.py code can be compute the physical distance (projected) between galaxies and the nearest filament detected.

Finally, it is possible to perform an analysis of the distribution of color, stellar mass, red and blue galaxy fractions, etc.,
as a function of the distance to the nearest filament.

The input must be a .csv file with a header specifying each column (e.g., RA_J2000,Dec_J2000,r_mag,g_mag,i_mag,z_phot).


