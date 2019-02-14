

import os
import time
import pickle
import numpy    as np
import scipy.io as io
import datetime as dt

from mayavi import mlab as mlab

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

def find_file(year,month,day,isObsPath=False,isObsFile=False ):
    if isObsPath : data_path = '/home/john/gDrive/Data/conde4D/Observation_Data'
    else         : data_path = '/home/john/gDrive/Data/conde4D'

    path = 0
    date = '{:0>4}-{:0>2}-{:0>2}'.format(year,month,day)
    #print(date)
    for parent,children,files in os.walk(data_path):
        for file in files:
            if date in file:
                if isObsPath:
                    if isObsFile and not('observations' in file):
                        continue
                    if not isObsFile and not('output' in file):
                        continue
                path = data_path + '/' + file
    return path

def normal_2_julian_secs(normal_times):
    try              : jtimes = [time.mktime(ti.timetuple()) for ti in normal_times]
    except TypeError : jtimes =  time.mktime(normal_times.timetuple())
    return np.array(jtimes)

#convert julian time into datetime objects
def julian_seconds_2_normal_time(jseconds):
    julian_epoch = dt.datetime(2000,1,1,hour=0)
    try:
        normal_date  = [julian_epoch + dt.timedelta(0,js) for js in jseconds]
        normal_date  = [event.replace(microsecond=0) for event in normal_date]
    except TypeError:
        normal_date  = julian_epoch + dt.timedelta(0,jseconds)
        normal_date  = normal_date.replace(microsecond=0)
    return normal_date

def get_obs_dataset(year,month,day):
    def retreive(tag,nc_file):
        return nc_file.variables[tag].data.copy()

    path    = find_file(year,month,day,True,True)
    windNC  = io.netcdf_file(path)

    times   = julian_seconds_2_normal_time(retreive('time',windNC))

    lons    = retreive('lon',windNC) - 360
    lats    = retreive('lat',windNC)
    alts    = retreive('alt',windNC)

    zones   = retreive('zon',windNC)
    meris   = retreive('mer',windNC)
    verts   = np.zeros_like(meris)

    uveczon = retreive('uveczon',windNC)
    uvecmer = retreive('uvecmer',windNC)
    uvecver = retreive('uvecver',windNC)

    loswind = retreive('loswind',windNC)
    siglos  = retreive('siglos' ,windNC)


    positions    = np.array( [   lons,   lats,   alts])
    initial_vels = np.array( [  zones,  meris,  verts])
    unit_vecs    = np.array( [uveczon,uvecmer,uvecver])

    return times,positions,initial_vels,unit_vecs,loswind,siglos



def get_conde_dataset(year,month,day,isObsPath=False,isObsFile=False):

    path = find_file(year,month,day,isObsPath,isObsFile)

    if not path:
        print("Failed to find file")
        return -1

    dataset = io.netcdf.netcdf_file(path)

    #Preklim data needed
    alts       = dataset.variables['alt' ].data.astype(float)
    times      = dataset.variables['time'].data.astype(float)

    #Get time breaks
    dtime     = np.gradient(times)
    dt_changes = np.where(dtime>0)[0]
    elements_per_step   = dt_changes[1]
    timesteps           = int(times.size/elements_per_step)


    #Get altititude breaks
    layers         = len(np.unique(alts))
    dalts          = np.gradient(alts)
    edges          = 2 #two sides to a gradient zero
    layer_idxs     = np.where(dalts>0)[0][1:layers*edges:edges]
    layer_idxs[-1] = 0
    layer_idxs.sort()


    #Helper for gettting bulk of data
    def retreive( key,layer_idxs,timesteps,elements_per_step ):
        data = dataset.variables[key].data.copy()
        if key=='lon':
            data -= 360


        if not (isObsPath or isObsFile):
            data = data.reshape(timesteps,elements_per_step).astype(float)
            layered_data = []
            for layer in range(layers):
                low_idx  =  layer_idxs[layer]

                if layer==(layers-1) : high_idx = -1
                else                 : high_idx = layer_idxs[layer+1]

                #print(low_idx,high_idx)
                layer_data = data[:,low_idx:high_idx]


                layered_data.append(layer_data)
            return layered_data
        else:
            return data


    lons  = retreive('lon',layer_idxs,timesteps,elements_per_step)
    lats  = retreive('lat',layer_idxs,timesteps,elements_per_step)
    alts  = retreive('alt',layer_idxs,timesteps,elements_per_step)
    zones = retreive('zon',layer_idxs,timesteps,elements_per_step)
    meris = retreive('mer',layer_idxs,timesteps,elements_per_step)
    verts = retreive('ver',layer_idxs,timesteps,elements_per_step)


    rot   = np.deg2rad(23.5)

    if not (isObsPath or isObsFile):
        mag_zones =  [ zones[layer]*np.cos(rot) + meris[layer]*np.sin(rot) for layer in range(layers)]
        mag_meris =  [-zones[layer]*np.sin(rot) + meris[layer]*np.cos(rot) for layer in range(layers)]
        time_grid = julian_seconds_2_normal_time(np.unique(times))
    else:
        mag_zones =  zones*np.cos(rot) + meris*np.sin(rot)
        mag_meris = -zones*np.sin(rot) + meris*np.cos(rot)
        time_grid = julian_seconds_2_normal_time(times)


    return lons,lats,alts,mag_zones,mag_meris,verts,np.array(time_grid)


def regularize_layered(data,times,time_grid):
    layers = len(data)

    regularized_data = []
    for layer in range(layers):
        reg_data = regularize(data[layer],times,time_grid)
        regularized_data.append(reg_data)

    return regularized_data

def regularize(data,times,time_grid):
    regular_data = np.zeros( (len(time_grid),data[0].size))
    #print(data.shape)
    for index,time in enumerate(time_grid):

        highs_dt = [timesi - time   for timesi in times]
        lows_dt  = [time   - timesi for timesi in times]

        highs    = np.array([ (-1)**(2+a.days) *a.seconds for a in highs_dt]).astype(float)
        lows     = np.array([ (-1)**(2+a.days) *a.seconds for a in  lows_dt]).astype(float)


        highs[highs<0] = 1e19
        lows [lows <0] = 1e19

        high_index = np.argmin(highs)
        lows_index = np.argmin( lows)

        total_distance  = times[high_index] - times[lows_index]
        travel_distance =              time - times[lows_index]

        if total_distance.seconds == 0.0:
            alpha = 1.0
            beta  = 0.0
        else:
            beta = travel_distance.seconds / total_distance.seconds
            alpha  = 1.0 - beta


        if travel_distance.seconds > 10*60:
            #print("SKIPPING")
            blended_image = np.zeros_like(data[0])
        else:
            #print(travel_distance, total_distance,time, times[lows_index],times[high_index],int(100*alpha),int(100*beta))
            blended_image = alpha * data[lows_index] + beta * data[high_index]

        #print(blended_image.mean(),data[lows_index].mean(),data[high_index].mean())
        regular_data[index] = blended_image

    return regular_data


def get_regular_data(year,month,day,time_grid):
    lons,lats,alts,zones,meris,verts,time_grid_raw = get_conde_dataset(year,month,day)

    data_list = lons,lats,alts,zones,meris,verts
    layers    = len(lons)

    processed_data = []
    for data in data_list:
        regular_data = []
        for layer in range(layers):
            regular_layer = regularize(data[layer],time_grid_raw,time_grid)
            regular_data.append(regular_layer)

        processed_data.append(regular_data)

    return processed_data

def get_regular_johnde4d_data(year,month,day,time_grid,isL2,isDivFreeish,extra_tag='',returnLOS=False):
    date = "{:0>4}-{:0>2}-{:0>2}".format(year,month,day)
    filename    = 'wind'+('_l1','_l2')[isL2] + ('','_divfree')[isDivFreeish]+f'{extra_tag}.pickle'
    pickle_path = f'/home/john/gDrive/Data/Reconstructions/{date}/Evolve/{filename}'
    with open(pickle_path,'rb') as pickle_file:
        data = pickle.load(pickle_file)

    jd_times            = data['recon_times']
    ts                  = normal_2_julian_secs(jd_times)
    dtime               = np.gradient(ts)
    dt_changes          = np.where(dtime>0)[0]
    elements_per_step   = dt_changes[1]
    timesteps           = int(ts.size/elements_per_step)

    jd_lons,jd_lats,jd_alts = data['recon_position']
    #Get layer breaks
    layers         = len(np.unique(jd_alts))
    dalts          =   np.gradient(jd_alts)
    edges          = 2 #two sides to a gradient zero
    layer_idxs     = np.where(dalts>0)[0][1:layers*edges:edges]
    layer_idxs[-1] = 0
    layer_idxs.sort()

    u,v,w = data['recon_wind_now']

    lons,lats,alts = [layer_data(pos,timesteps,elements_per_step,layers,layer_idxs)\
                                                        for pos in [jd_lons,jd_lats,jd_alts] ]
    U,V,W = [layer_data(wind_component,timesteps,elements_per_step,layers,layer_idxs)\
                                                        for wind_component in [u,v,w] ]
    times_layered = layer_data(jd_times,timesteps,elements_per_step,layers,layer_idxs)

    data['recon_layered_pos']  = [lons,lats,alts]
    data['recon_layered_wind'] = [U,V,W]
    data['recon_layered_times']= times_layered

    jd_times                = np.unique(jd_times)
    jd_lons,jd_lats,jd_alts = data['recon_layered_pos']
    jd_lons,jd_lats,jd_alts = [regularize_layered(mat,jd_times,time_grid) for mat in [jd_lons,jd_lats,jd_alts] ]
    jd_U,jd_V,jd_W          = data['recon_layered_wind']
    jd_U,jd_V,jd_W          = [regularize_layered(mat,jd_times,time_grid) for mat in [jd_U,jd_V,jd_W] ]

    if returnLOS:
        return jd_lons,jd_lats,jd_alts,jd_U,jd_V,jd_W,
    else:
        return jd_lons,jd_lats,jd_alts,jd_U,jd_V,jd_W


if __name__ == '__main__':

    year,month,day = 2017,3,2

    data = get_conde_dataset(year,month,day,True)
    lons,lats,alts,zones,meris,verts,time_grid = data#[np.hstack(mat) for mat in data]
#
#    uni_lons = np.unique(lons)
#    uni_lats = np.unique(lats)
#    uni_alts = np.unique(alts)
#
#    #layers    = len(lons)
#    timesteps = len(lons)


def fake():
    lon_3d,lat_3d,alt_3d = np.meshgrid(uni_lons,uni_lats,uni_alts)

    U,V,W = np.zeros((3,timesteps,uni_lats.size,uni_lons.size,uni_alts.size))

    for index in range(lons[0].size):
        i = np.where(uni_lats==lats[0][index])
        j = np.where(uni_lons==lons[0][index])
        k = np.where(uni_alts==alts[0][index])

        U[:,i,j,k] = zones[:,index].reshape(timesteps,1,1)
        V[:,i,j,k] = meris[:,index].reshape(timesteps,1,1)
        W[:,i,j,k] = verts[:,index].reshape(timesteps,1,1)



    def get_instance(idx):

        lon   = 10*lons [idx]
        lat   = 10*lats [idx]
        alt   = alts [idx]
        zone  = zones[idx]
        meri  = meris[idx]
        vert  = verts[idx]
        time  = time_grid[idx]

        return lon ,lat ,alt ,\
               zone,meri,vert,\
               time


    #vectors = []
    #for layer in range(layers):
    x,y,z,u,v,w,t = get_instance(0)
    vectors = mlab.quiver3d(x,y,z,u,v,w)
    #    vectors.append(layer_vectors)


    @mlab.animate(delay=75,ui=True)
    def animate():
        while True:
            for idx in range(timesteps):
                date      = time_grid[idx].date()
                time      = time_grid[idx].time().isoformat()
                title     = "Viewing wind for the night of {} at {} UT".format(date, time)

                text      = mlab.title(title, color=(0.75, 0.75, 0.75))
                text.property.font_family            = 'times'
                text.property.justification          = 'centered'
                text.property.vertical_justification = 'centered'
                text.property.shadow                 = True
                text.property.use_tight_bounding_box = True
                text.property.opacity                 = 0.7429


                #for layer,layer_vectors in enumerate(vectors):
                x,y,z,u,v,w,t = get_instance(idx)
                vectors.mlab_source.set(u=u,v=v,w=w)

                yield

    animate()
