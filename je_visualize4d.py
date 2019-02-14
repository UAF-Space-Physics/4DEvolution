# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 20:46:45 2018

@author: John
"""

import os
import time
import scipy  as np
import pickle
import mayavi.mlab as mlab

layer_mask = 1,0,0,0,0,1

#pickle_path = '/home/john/gDrive/Data/Reconstructions/2017-03-02/Evolve/wind.pickle'
pickle_path = '/home/john/gDrive/Data/Reconstructions/2017-03-02/Evolve/wind.pickle'

def normal_2_julian_secs(normal_times):
    try              : jtimes = [time.mktime(ti.timetuple()) for ti in normal_times]
    except TypeError : jtimes =  time.mktime(normal_times.timetuple())
    return np.array(jtimes)

def layer_data(input_data,timesteps,elements_per_step,layers,layer_idxs):
    data = input_data.reshape(timesteps,elements_per_step)
    layered_data = []
    for layer in range(layers):
        low_idx  =  layer_idxs[layer]
        if layer==(layers-1) : high_idx = -1
        else                 : high_idx = layer_idxs[layer+1]
        layer_data = data[:,low_idx:high_idx]
        layered_data.append(layer_data)
    return layered_data

#Nice helper to turn lats and lons into cartesian coords on a sphere
def latlon3D(lon,lat,height):
    if type(lon) == list:
        lon = np.array(lon)
    if type(lat) == list:
        lat = np.array(lat)
    if type(height) == list:
        height = np.array(height)

    phi   = np.deg2rad(lon)
    theta = np.deg2rad(np.absolute(90 - lat))
    R     = 6350 + height #km

    x  = R*np.sin(theta)*np.cos(phi)
    y  = R*np.sin(theta)*np.sin(phi)
    z  = R*np.cos(theta)

    return x,y,z



if __name__ == '__main__':
    with open(pickle_path,'rb') as pickle_file:
        data = pickle.load(pickle_file)
#    with open(pickle_path2,'rb') as pickle_file:
#        data2 = pickle.load(pickle_file)

    times          = data['recon_times']
    lons,lats,alts = data['recon_position']
    #Get time breaks
    ts                  = normal_2_julian_secs(times)
    dtime               = np.gradient(ts)
    dt_changes          = np.where(dtime>0)[0]
    elements_per_step   = dt_changes[1]
    timesteps           = int(ts.size/elements_per_step)

    #Get layer breaks
    layers         = len(np.unique(alts))
    dalts          = np.gradient(alts)
    edges          = 2 #two sides to a gradient zero
    layer_idxs     = np.where(dalts>0)[0][1:layers*edges:edges]
    layer_idxs[-1] = 0
    layer_idxs.sort()

    u,v,w = data['recon_wind_now']

    lons,lats,alts = [layer_data(pos,timesteps,elements_per_step,layers,layer_idxs)\
                                                        for pos in [lons,lats,alts] ]
    U,V,W = [layer_data(wind_component,timesteps,elements_per_step,layers,layer_idxs)\
                                                        for wind_component in [u,v,w] ]
    times_layered = layer_data(times,timesteps,elements_per_step,layers,layer_idxs)

    data['recon_layered_pos']  = [lons,lats,alts]
    data['recon_layered_wind'] = [U,V,W]
    data['recon_layered_times']= times_layered
    with open(pickle_path,'wb') as pickle_file:
        pickle.dump(data,pickle_file,protocol=-1)

    exit()

    ####################################################################################################################
    # Create the Earth
    ####################################################################################################################

    #Define the radius of our planet
    R = 6351


    from tvtk.api import tvtk
    if os.name=='nt':
        img_filename = 'ETOPO1_reduced.jpg'
    else:
        img_filename = '/home/john/gDrive/Code/Python/Research/GreenRedComparison/ETOPO1.jpg'
    #img          =  np.array( Image.open(img_filename) ).mean(axis=2)
    #texture      = mlab.pipeline.array2d_source(img)

    img          = tvtk.JPEGReader(file_name=img_filename)
    texture      = tvtk.Texture(input_connection=img.output_port, interpolate=1)

    sphere = mlab.points3d(0, 0, 0, \
                           scale_mode   ='none',
                           color        =(0.5,0.5,0.5),
                           scale_factor = 2*R  ,
                           resolution   = 50   ,
                           opacity      = 0.55  ,
                           name         ='Earth')


    # These parameters, as well as the color, where tweaked through the GUI,
    # with the record mode to produce lines of code usable in a script.
    sphere.actor.property.specular       = 0.45
    sphere.actor.property.specular_power = 5
    # Backface culling is necessary for more a beautiful transparent
    # rendering.
    sphere.actor.tcoord_generator_mode         = 'sphere'
    sphere.actor.property.backface_culling     = True
    sphere.actor.mapper.scalar_visibility      = False
    sphere.actor.enable_texture                = True
    #sphere.actor.texture_source_object         = texture
    sphere.actor.texture                       = texture
    sphere.actor.tcoord_generator.prevent_seam = False

    ####################################################################################################################
    # Put cities on this Earth
    ####################################################################################################################

    city_lats  = [  68.63       ,  65.13        ,  70.13    ,   64.84       ,  61.22        ,   62.3   ,  64.8   ,   66.6 ]
    city_lons  = [-149.60       ,-147.48        , -143.64   , -147.72       ,-149.90        , -145.3   , -141.2  , -145.3 ]
    city_names = ['Toolik Lake' , 'Poker Flat' , 'Kaktovik', 'Fairbanks'   , 'Anchorage'   , 'Gakona' , 'Eagle' , 'Fort Yukon']

    x_city,y_city,z_city = latlon3D(city_lons,city_lats,0.5)
    mlab.points3d(x_city,y_city,z_city,scale_factor=1)
    for index,label in enumerate(city_names):
        mlab.text3d(x_city[index],y_city[index],z_city[index], label, scale=(5, 5, 5))

    vectors = []

    for layer in range(layers):
        idx   = 0
        if layer_mask[layer]:
            x,y,z = latlon3D(lons[layer][idx],lats[layer][idx],alts[layer][idx])
            u,v,w = U[layer][idx],V[layer][idx],W[layer][idx]
            field = mlab.quiver3d(x,y,z,u,v,w)
        else:
            field = 0
        vectors.append(field)

    @mlab.animate(delay=100)
    def anim(idx=0):
        instances = len(U[0])
        while True:
            idx = idx%instances
            for layer in range(layers):
                if layer_mask[layer]:
                    x,y,z = latlon3D(lons[layer][idx],lats[layer][idx],alts[layer][idx])
                    u,v,w = U[layer][idx],V[layer][idx],W[layer][idx]
                    vectors[layer].mlab_source.reset(x=x,y=y,z=z,u=u,v=v,w=w)
            idx +=1
            yield

    anim()
    mlab.show()
