import math
import astropy.table as tbd
import astropy.cosmology as cc
import os
import numpy as np

def distance_point_to_line_segment(point, start, end):
    # Calculate the distance from a point to a line segment
    x, y = point
    x1, y1 = start
    x2, y2 = end
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if len_sq != 0:
        param = dot / len_sq

    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = x - xx
    dy = y - yy
    return math.sqrt(dx * dx + dy * dy)

def nearest_line_segment(point, start, end):
    #counter = 0
    #cn = 0
    # Find the nearest line segment from an array of line segments to a point
    min_distance = float('inf')
    nearest_segment_st = None
    nearest_segment_ed = None
    for i,st in enumerate(start):
        #print(st,end[i])
        distance = distance_point_to_line_segment(point, st,end[i])
        #cn += 1
        if distance < min_distance:
            min_distance = distance
            nearest_segment_st = st
            nearest_segment_ed = end[i]
            #counter += 1
        #print(cn,counter)
    return  min_distance,nearest_segment_st,nearest_segment_ed


def filcalc(segfl,galcat,z_slice,ra,dec,z=None):

    U0,U1,V0,V1 = np.loadtxt(segfl, usecols=(0,1,2,3), comments='#', unpack=True)

    U0_U1R = np.sqrt(np.power(U0,2)+np.power(U1,2))
    V0_V1R = np.sqrt(np.power(V0,2)+np.power(V1,2))

    segs_new = tbd.Table()

    segs_new['U0'] = U0
    segs_new['U1'] = U1
    segs_new['V0'] = V0
    segs_new['V1'] = V1
    segs_new['U0_U1R'] = U0_U1R
    segs_new['V0_V1R'] = V0_V1R

    
    ## Calculating distances in Mpc and saving to output file ##
    

    kpc_scale = 1.241#cc.cosmocalc(Z=z_slice,hc=67.4,wm=0.315,wv=0.685)[2]

    mpc_scale = 3.6*kpc_scale
    U0_U1Rmpc = U0_U1R* mpc_scale
    V0_V1Rmpc = V0_V1R* mpc_scale
    segs_new['U0_U1Rmpc']=U0_U1Rmpc
    segs_new['V0_V1Rmpc']=V0_V1Rmpc

    segs_new.write(segfl+'.csv',overwrite=True)


    tst = np.vstack((U0,U1)).T
    tst1 = np.vstack((V0,V1)).T

    st_nearsegs = []
    ed_nearsegs = []
    dist_neardeg = np.array([])

    for i,ras in enumerate(ra):
        dists,sts,eds, = nearest_line_segment((ras,dec[i]),tst,tst1)
        st_nearsegs.append(sts)
        ed_nearsegs.append(eds)
        dist_neardeg = np.append(dist_neardeg,dists)

    distmpc = dist_neardeg*mpc_scale

    st_x=np.array([])
    st_y=np.array([])
    ed_x =np.array([])
    ed_y= np.array([])

    for ss in st_nearsegs:
        st_x = np.append(st_x,ss[0])
        st_y = np.append(st_y,ss[1])
    
    for ee in ed_nearsegs:
        ed_x = np.append(ed_x,ee[0])
        ed_y = np.append(ed_y,ee[1])
    

    galpath,galfile=os.path.split(galcat)

    galname,galext=os.path.splitext(galfile)

    nearesttab = tbd.Table()   

    nearesttab['RA'] = ra
    nearesttab['DEC'] = dec
    nearesttab['start_x'] = st_x
    nearesttab['start_y'] = st_y
    nearesttab['end_x'] = ed_x
    nearesttab['end_y'] = ed_y
    nearesttab['distseg'] = dist_neardeg
    nearesttab['distmpc'] = distmpc


    nearesttab.write(galpath+'/'+galname+'_DistCals.csv')

    return

