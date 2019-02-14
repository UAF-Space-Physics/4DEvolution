# -*- coding: utf-8 -*-
"""
Created on Sun May 27 16:27:29 2018

@author: John Elliott
"""

'''*****************************************************************************
* Import Functionality
*****************************************************************************'''
import gc       # Garbage Collection
import os       # OS routines
import time     # Time routines
import pickle   # Pickling routines

#Import library to handle Conde input data
import lib.conde4d as c4

#Bring in math functionality
from   scipy  import io   as io
import scipy              as np
import torch              as th

#Plotting routines
import pylab              as plt
import matplotlib.dates   as mdates

#plt.style.use('dark_background')

'''*****************************************************************************
* Runtime Paramters
*****************************************************************************'''
if __name__ == '__main__':

    year,month,day = 2017,3,2

    #Are we plotting anything?
    isPlotting     = 1
    stride         = 1

    #Are we resuming a run? No.
    isResuming     = 0 #Not working

    #How many iterations to run and are they divergence free iterations with an L2 loss?
    imax           = int(20e6)
    isDivFreeish   = False
    isL2           = True
    isSmartScaling = False

    #Nominal horizontal and temporal std in perturbations
    h_sigma =    1.0 #degrees
    z_sigma =     50 #km altitude
    t_sigma =    600 #secs

    #Update rates for various triggers
    hf_rate = 10000
    lf_rate = 25000

    #The loss function needs a nominal speed to regularize
    max_desired_speed = np.array( [300.0,300.0,10.0] )

#Helper for erfs -- needed an explicit function
a_for_erf = 8.0/(3.0*np.pi)*(np.pi-3.0)/(4.0-np.pi)
def erf_approx(x):
    return th.sign(x)*th.sqrt(1-th.exp(-x*x*(4/np.pi+a_for_erf*x*x)/(1+a_for_erf*x*x)))

#Helper function for CUDA tensors
def T(mat) :
    retMat = th.from_numpy(mat.astype(np.float32)).cuda()
    return retMat

#Helper function to convert
def normal_2_julian_secs(normal_times):
    try              : jtimes = [time.mktime(ti.timetuple()) for ti in normal_times]
    except TypeError : jtimes =  time.mktime(normal_times.timetuple())
    ntimes = np.array(jtimes)
    return ntimes

#Helper for Random numbers
def scale_random (maxr,minr=0,type=int) :
    scale = (maxr - minr)
    rand  = th.rand(1,device='cuda:0')
    return type( scale * rand + minr)

#Helper for Gaussians, does both observation and reconstruction grids together
def gaussian(x1,x2,x0,sigma,index_ratio=0):
    numer1    = (x1 - x0)**2
    numer2    = (x2 - x0)**2

    denom     =   2*sigma**2
    scale     = 1 #(2*np.pi*sigma**2)**(-0.5)

    g1        = scale*th.exp(-numer1/denom)
    g2        = scale*th.exp(-numer2/denom)
    return g1,g2

#Helper for altitude-dependent Gaussians -- Not nearly the same as above, does both observation and reconstruction grids together
def height_corrected_gaussian(x1,x2,x0,sigma0,index_ratio=0):
    numer1    = (x1 - x0)**2
    numer2    = (x2 - x0)**2

    #Create height dependent sigmas
    sigma1    =  variable_sigma(x1,sigma0)
    sigma2    =  variable_sigma(x2,sigma0)
    #Feed into denoms
    denom1    = 2*sigma1**2
    denom2    = 2*sigma2**2

    scale1    = 1 #(2*np.pi*sigma1**2)**(-0.5)
    scale2    = 1 #(2*np.pi*sigma2**2)**(-0.5)

    g1        = scale1*th.exp(-numer1/denom1)
    g2        = scale2*th.exp(-numer2/denom2)
    return g1,g2

def variable_sigma(domain,sigma,max_mag=1):
    #varied_sigma =   (sigma - domain)**0.5
    result       = domain/10000 #max_mag*varied_sigma/varied_sigma.max()
    return result


#Make the x,y,z,t perturbations as needed
def gauss_pert(pert_axis,obs_pos,recon_pos,pert_pos,sigmas,ts1,ts2,pert_time,\
                                            t_sigma,index_ratio,base_z_sigma):
    #wrangle everything into uniform syntax
    function = gaussian,gaussian,gaussian
    scale    = scale_random(4, 0.5*(1-index_ratio),float)*sigmas[0],\
               scale_random(4, 0.5*(1-index_ratio),float)*sigmas[1],\
               base_z_sigma,\
               scale_random(4,0.5,float)*t_sigma


    #Calculate the perturbation in 4d, one d at a time
    xp,yp,zp = [function[i](obs_pos[i],recon_pos[i],pert_pos[i],scale[i],index_ratio)\
                                                        for i in range(len(pert_pos))]
    tp       = gaussian(ts1,ts2,pert_time,scale[-1],index_ratio)

    #Return the 4d product
    return xp,yp,zp,tp


def make_pert(obs_positions,recon_positions,ts1,ts2,index_ratio=0,base_z_sigma=80):
    ntimes   = recon_positions.shape[1]
    pert_idx = scale_random(ntimes)
    pert_pos = recon_positions[:,pert_idx]
    pert_axis= scale_random( len(pert_pos) )

    #Scale factor for the perturbation... no need to change these unless you
    #specifically know what you are wanting to alter
    mag      = scale_random(0.9,1.1) #**(2-index_ratio*2)

    xp,yp,zp,tp = gauss_pert(pert_axis,obs_positions,recon_positions,pert_pos,vel_sigma,\
                                ts1,ts2,ts2[pert_idx],t_sigma,index_ratio,base_z_sigma)

    p_obs,p_recon = [mag*xp[i]*yp[i]*zp[i]*tp[i] for i in range(len(xp))]

    #Create a container for the pert and populate it accordingly
    pert_obs  ,pert_recon = th.zeros_like(obs_positions),th.zeros_like(recon_positions)
    pert_obs  [pert_axis] = p_obs
    pert_recon[pert_axis] = p_recon

    #If we're rocking divfree perts make them divfree
    if isDivFreeish:
        if pert_axis == 2: #Less compact form is more insightful here
            nox_pert = [mag*yp[i]*zp[i]*tp[i] for i in range(len(xp))]
            noy_pert = [mag*xp[i]*zp[i]*tp[i] for i in range(len(xp))]

            pert_obs  [0] = nox_pert[0]*divfree_mag(pert_axis,0,  obs_positions,pert_pos,base_z_sigma)/2
            pert_recon[0] = nox_pert[1]*divfree_mag(pert_axis,0,recon_positions,pert_pos,base_z_sigma)/2
            pert_obs  [1] = noy_pert[0]*divfree_mag(pert_axis,1,  obs_positions,pert_pos,base_z_sigma)/2
            pert_recon[1] = noy_pert[1]*divfree_mag(pert_axis,1,recon_positions,pert_pos,base_z_sigma)/2
        else:
            other_axis             = int(not pert_axis)
            noz_pert               = [mag*xp[i]*yp[i]*tp[i] for i in range(len(xp))]
            pert_obs  [other_axis] = noz_pert[0]*divfree_mag(pert_axis,other_axis,  obs_positions,pert_pos,base_z_sigma)
            pert_recon[other_axis] = noz_pert[1]*divfree_mag(pert_axis,other_axis,recon_positions,pert_pos,base_z_sigma)
    return pert_obs,pert_recon

#Divfree calc function
def divfree_mag(pert_axis,other_axis,positions,pert_pos,base_z_sigma):
    dx     = positions[pert_axis]  - pert_pos[pert_axis]
    dy     = positions[other_axis] - pert_pos[other_axis]

    if pert_axis==2 : sigma = variable_sigma(positions[pert_axis],base_z_sigma)
    else            : sigma = vel_sigma[pert_axis]

    mag1   = erf_approx( dy/vel_sigma[other_axis] )
    mag2   = vel_sigma[other_axis]*(np.pi**0.5)*(-dx)/sigma**2
    mag2  *= scale_random(1.1,0.9,float)

    return mag1*mag2

#Cost function
def goodness_gracious_l2_norms_of_fire(obs_loswind,obs_recon_vels,obs_unit_vecs,\
                                                        uncertainty,alpha=0.0001):
    recon_los        = th.sum( obs_recon_vels*obs_unit_vecs,0 )
    if isL2 : L      = th.sum((      (recon_los - obs_loswind)**2)/uncertainty )
    else    : L      = th.sum((th.abs(recon_los - obs_loswind)   )/uncertainty )
    L                = L/recon_los.size()[0]
    bigwind_penalty  = th.sum( (th.abs(obs_recon_vels).t()/max_desired_speed) )/(obs_recon_vels.size()[0])
    loss             = (L + alpha*bigwind_penalty).cpu().numpy()
    return loss


'''*****************************************************************************
* Main Loop
*****************************************************************************'''
if __name__ == '__main__':
    #These should be left alone
    vel_sigma       = h_sigma,h_sigma,z_sigma
    extra_tag       = ''
    qaulifications  = ('','DivFree ')[isDivFreeish] + ('L1','L2')[isL2]

    max_desired_speed = T(max_desired_speed)

    #automatic boilerplate -- Probably want to leave this be
    save_folder = '/home/john/gDrive/Data/Reconstructions/{:0>4}-{:0>2}-{:0>2}/Evolve/'.format(year,month,day)
    try                    : os.mkdir(save_folder)
    except FileExistsError : pass #quietly into the night
    save_path = save_folder + 'wind'+('_l1','_l2')[isL2] + ('','_divfree')[isDivFreeish]+f'{extra_tag}.pickle'

    #Grab the observation data
    obsgrid_data = c4.get_obs_dataset(year,month,day)
    obs_times    = obsgrid_data[0]
    obs_positions,obs_init_vels,obs_unit_vecs,obs_loswind,obs_siglos = \
                                    [T(mat) for mat in obsgrid_data[1:] ]

    #Grab the recon data
    recongrid_data = c4.get_conde_dataset(year,month,day,True,False)
    recon_times    = recongrid_data[-1]
    recon_lons,recon_lats,recon_alts,recon_zones,recon_meris,recon_verts = \
                                    [T(mat) for mat in recongrid_data[:-1] ]

    #Stack it on the GPU
    recon_positions = th.stack([recon_lons,recon_lats,recon_alts])

    #Convert the times and transfer to GPU
    ts1  = T(normal_2_julian_secs(obs_times))
    ts2  = T(normal_2_julian_secs(recon_times))

    #Do plotting if we are doing plotting, as they say
    if isPlotting:
        plt.ion()
        windfig,windax = plt.subplots(nrows=3,ncols=1)
        L2fig,L2ax     = plt.subplots(nrows=3,ncols=1)
        myFmt          = mdates.DateFormatter(' %H:%M ')

    #Resume if we are resuming. Otherwise set the initial arrays on the GPU
    if isResuming:
        with open(save_path,'rb') as pickle_file:
            data = pickle.load(pickle_file)
        recon_recon_wind = T(data['recon_wind_now'])
        obs_recon_wind   = T(data  ['obs_wind_now'])
    else:
        obs_recon_wind   = obs_init_vels.clone()
        recon_recon_wind = th.stack([recon_zones,recon_meris,recon_verts])
        recon_init_vels  = recon_recon_wind.clone()

    #All the arrays need initial values
    L2_0            = goodness_gracious_l2_norms_of_fire(obs_loswind,obs_init_vels,\
                                                         obs_unit_vecs,obs_siglos)
    L2              = [L2_0]
    current_L2      = L2_0
    L2_prog_idx     = [0]
    L2_progress     = [L2_0]
    accepted_perts  = 0
    acceptances     = [100]
    acceptance_idx  = [0]

    #Set the perturbation and fitting guidelines
    max_mag         = 1
    mag_test_points = 5
    mags            = []
    #Don't edit v
    flexiscale      = 1
    test_range      = T(np.linspace(-max_mag,max_mag,mag_test_points))


    #More initialization if we're not resuming
    if not isResuming:
        data                         = {}
        data  ['recon_wind_initial'] = recon_init_vels.clone().cpu().numpy()
        data      ['recon_position'] = recon_positions.clone().cpu().numpy()
        data    ['obs_wind_initial'] =  obs_recon_wind.clone().cpu().numpy()
        data        ['obs_position'] =   obs_positions.clone().cpu().numpy()
        data         ['recon_times'] = recon_times
        data           ['obs_times'] = obs_times

    #Start a clock and get going
    tStart = time.time()
    for i in range(imax):
        #Make the perts and test them for goodness of fit
        index_ratio   = min([0.99,i**(1.01)/imax])
        p_obs,p_recon = make_pert(obs_positions,recon_positions,ts1,ts2,index_ratio=index_ratio)
        perts         = [p_obs*testi for testi in test_range]
        L2_i_tests    = [goodness_gracious_l2_norms_of_fire(obs_loswind,obs_recon_wind+pert,obs_unit_vecs,obs_siglos) for pert in perts]

        #With the GOFs fit a polynomial that would represent the most optimal magnitude fo pert
        coeffs        = np.polyfit(test_range,L2_i_tests,2)

        if isSmartScaling: #It's not smart... just sayin'
            der           = np.polyder(coeffs)
            roots         = np.roots(der)
            roots         = roots[np.isreal(roots)].real
            vals          = np.polyval(coeffs,roots)
            mag           = roots[np.argmin(vals)]

            signless_mag  = np.absolute(mag)
            if signless_mag > max_mag:
                mag = max_mag*mag/signless_mag
        else: #This is the simple way. Just sample a distribution over a range and eval
            x = np.random.normal(size=1000)*max_mag
            y = np.polyval(coeffs,x)
            minimag = np.argmin(y)
            mag = x[minimag]

        #Add the appropriate pert and evaluate whether we improved the situation
        perty_wind = obs_recon_wind+p_obs*mag
        L2_i       = goodness_gracious_l2_norms_of_fire(obs_loswind,perty_wind,obs_unit_vecs,obs_siglos)


        #If we did improve the situation, add the pert to our wind field
        if L2_i < current_L2 :
            accepted_perts     +=  1
            obs_recon_wind  [:] = obs_recon_wind + p_obs*mag
            recon_recon_wind[:] = recon_recon_wind + p_recon*mag
            current_L2          = L2_i

            if isPlotting:
                L2_prog_idx.append(i)
                L2_progress.append(current_L2)
        else:
            mag = 0.0
        #If we're plotting, save some values to plot
        if isPlotting:
            L2.append(L2_i)
            mags.append(mag)

        if i:
            if not i%hf_rate:
                kiters_done     =  i/1e3
                iterrate        = int(60 * hf_rate / (time.time() - tStart))
                tStart          = time.time()

                acceptance_rate = accepted_perts/1e2
                accepted_perts  = 0

                if isPlotting:
                    acceptance_idx.append(i)
                    acceptances.append(acceptance_rate)
                    acceptances[0] = acceptances[1] #hack! The assignment is quicker than logically checking

                print(f"{kiters_done}k iters \t:\t{qaulifications} Cost {current_L2:.4f} \t:\t {iterrate} iters/min \t:\t Pert acceptance rate {acceptance_rate}%")

            if not i%lf_rate:
                data['iterations_performed'] = i
                data      ['recon_wind_now'] = recon_recon_wind.cpu().numpy()
                data        ['obs_wind_now'] =   obs_recon_wind.cpu().numpy()
                with open(save_path,'wb') as pickle_file:
                    pickle.dump(data,pickle_file,protocol=-1)
                gc.collect()

            if not (i-1)%hf_rate:
                if isPlotting:
                    for ax in [*windax,*L2ax]:
                        ax.clear()

                    L2ax[0].semilogy(L2_prog_idx,L2_progress,marker='.')
                    L2ax[1].scatter (range(i+1),mags,s=0.1)
                    L2ax[2].plot    (acceptance_idx,acceptances)

                    title = f"{('L1','L2')[isL2]} Cost"
                    L2ax[0].set_title(title)
                    L2ax[1].set_title("Relative Magnification")
                    L2ax[2].set_title("Pert Acceptance Rate")


                    for comp_idx,comp in enumerate(recon_recon_wind):
                        windax[comp_idx].scatter(recon_times    [::10],\
                          recon_init_vels[comp_idx][::10],alpha=0.5,color='blue',s=0.1)
                        L2fig.suptitle  (f'Iteration: {i} of {imax}')
                        windax[comp_idx].scatter(recon_times[::stride],comp[::stride],\
                                       alpha=0.7,c=recon_alts[::stride],s=0.05,cmap='RdYlGn_r')

                    windfig.suptitle(f'Iteration: {i} of {imax}')

                    title = 'Zonal Winds','Meridional winds','Vertical Winds'
                    for direction,ax in enumerate(windax):
                        ax.xaxis.set_major_formatter(myFmt)
                        ax.set_xlim(recon_times[0],recon_times[-1])
                        ax.set_title(f"{title[direction]} (m/s)")

                    windfig.tight_layout()
                    L2fig.tight_layout()
                    L2fig.canvas.draw  ()
                    windfig.canvas.draw()

                    plt.pause(0.001)
