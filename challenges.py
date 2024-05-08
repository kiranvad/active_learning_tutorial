from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern
from scipy.signal import savgol_filter
import xarray as xr
xr.set_options(display_expand_data=False)
from tutlib import *
from tutlib.instruments.tutorial import *
from tutlib.instruments import starting_composition_list
from tutlib.instruments.challenge1 import get_virtual_instrument as get_virtual_instrument1
from tutlib.instruments.challenge2 import get_virtual_instrument as get_virtual_instrument2
from tutlib.instruments.challenge3 import get_virtual_instrument as get_virtual_instrument3
import numpy as np
np.random.seed(240424)
import scipy.interpolate as interp
from autophasemap import multi_kmeans_run
import os, shutil, sys
import ray

CHALLENGE = int(sys.argv[1])
data_loc_names = ["challenge1.nc", "challenge2v3.nc", "challenge1.nc"]
num_phases = [2,4,2]
spline_s_values = [1.0, 0.05, 0.025]

if not "ip_head" in os.environ:
    ray.init()
else:
    ray.init(address='auto', 
             _node_ip_address=os.environ["ip_head"].split(":")[0],
             _redis_password=os.environ["redis_password"]
             )

num_nodes = len(ray.nodes())
print('Total number of nodes are {}'.format(num_nodes))
print('Avaiable CPUS : ', ray.available_resources()['CPU'])

class DataSet:
  def __init__(self, C, q, Iq):
    self.flags = (Iq>0.001).all(axis=0)
    self.q = q[self.flags]
    self.Iq = Iq[:,self.flags]
    self.C = C
    self.N = self.C.shape[0]
    self.n_domain = 200
    self.q_logspace = np.geomspace(self.q.min(), self.q.max(), self.n_domain)
    self.q_log = np.log10(self.q_logspace)
    self.t = (self.q_log-min(self.q_log))/(max(self.q_log)-min(self.q_log))

  def generate(self, spline_s = 0.5):
    self.F = []
    for i in range(self.N):
      spline = interp.splrep(np.log10(self.q), 
                             np.log10(self.Iq[i,:]), 
                             s=spline_s
                             )
      Iq = interp.splev(self.q_log, spline)

      self.F.append(Iq)
    
    return 


def label( dataset: xr.Dataset, num_phases: int) -> xr.Dataset:
    ''' A method which labels (classifies) the virtual scattering data

    Contract: Make the following changes to the 'dataset' variable
      - Set variables "labels" with dimension "sample"
      - Set attribute "n_phases"
    '''

    C = dataset[['a','b','c']].to_array().values.T
    data = DataSet(C, dataset["q"].values, dataset['sas'].values)
    data.generate(spline_s = spline_s_values[CHALLENGE-1])
    # create autophasemap clustering object
    clf,_ = multi_kmeans_run(10, 
                           data, 
                           num_phases, 
                           max_iter=30, 
                           verbose=3, 
                           smoothen=False
    )
    #store results in the dataset
    dataset['labels'] = ('sample', clf.delta_n)
    dataset['labels'].attrs['description'] = 'ML-defined label for each measurement'
    dataset.attrs['n_phases'] = num_phases

    return dataset

def extrapolate(dataset: xr.Dataset) -> xr.Dataset:
    ''' Extrapolate the labels from the previous step over a composition space

    Contract: Make the following changes to the 'dataset' variable
      - Set variable "labels_grid" with dimension "grid"
      - Set variable "labels_grid_prob" with dimensions ("grid","phase")
    '''
    # create and train a GP
    clf = GaussianProcessClassifier(
        kernel = Matern(
            length_scale_bounds=(0.05,0.25),
            nu=1.5
            ),
        )

    clf.fit(
        X=dataset[['c','a','b']].to_array().transpose('sample',...),
        y=dataset['labels']
    )

    # gather 'grid' of composition points to evaluate trained GP on
    Xp = dataset[['c_grid','a_grid','b_grid']].to_array().transpose('grid',...)

    # evaluate trained GP on grid
    labels_grid = clf.predict(Xp)

    # calculate entropy of prediction
    label_prob = clf.predict_proba(Xp)

    #store results in the dataset
    dataset['labels_grid_prob'] = (('grid','phase'), label_prob)
    dataset['labels_grid_prob'].attrs['description'] = 'Probability of each label at every composition of the grid'

    dataset['labels_grid'] = ('grid', labels_grid)
    dataset['labels_grid'].attrs['description'] = 'Most likely label at every composition of the grid'

    return dataset

def choose_next_acquisition(dataset: xr.Dataset) -> xr.Dataset:
    """Use information from previous methods to decide next optimal measurement

    Contract: Make the following changes to the 'dataset' variable
      - Set variables "acquisition" with dimension "grid"
      - add a dictionary named 'next_sample' to the attributes of dataset
      - next sample should be like {'a':0.1, 'b':0.2, 'c':0.7}
    """
    # calculate the acquisition surface from the label probability of the GP.
    # In this case, we'll calculate the entropy of the mean function.
    yp = dataset['labels_grid_prob']
    acquisition = (-np.log(yp)*yp).sum('phase')

    # get index max value of acquisition surface and cast to integer
    next_sample_index = int(acquisition.argmax())

    # get composition associated with selection
    next_sample_comp = dataset[['c_grid', 'a_grid', 'b_grid']].isel(grid = next_sample_index)

    next_sample_comp_dict = {k.replace('_grid',''):v for k,v in next_sample_comp.to_pandas().to_dict().items()}
    dataset.attrs['next_sample'] = next_sample_comp_dict

    dataset["acquisition"] = acquisition #don't need to specify dimension because it's already a DataArray
    dataset["acquisition"].attrs['description'] = "acquisition/decision surface"

    return dataset 

virtual_intruments = [get_virtual_instrument1, get_virtual_instrument2, get_virtual_instrument3]
SAVE_DIR = "./figures_V2/challenge_%d/"%(CHALLENGE)
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)

print('Saving the results to %s'%SAVE_DIR)


instrument = virtual_intruments[CHALLENGE-1](boundary_dataset_path = os.getcwd()+"/challenge_datasets/%s"%(data_loc_names[CHALLENGE-1]), 
                                reference_data_path=os.getcwd()+"/reference_sans/"
                                )

input_dataset = instrument.measure_multiple(starting_composition_list)
results = actively_learn(niter = 101,
                          num_phases=num_phases[CHALLENGE-1],
                          input_dataset=input_dataset,
                          label=label,
                          extrapolate=extrapolate,
                          choose_next_acquisition=choose_next_acquisition, 
                          instrument=instrument,
                          plot=True,plot_every=10,plot_skip_phases=['D'], plot_save_path=SAVE_DIR
                        )
results.attrs['next_sample'] = str(results.attrs['next_sample'])
results.to_netcdf(SAVE_DIR+'/challenge_%d.nc'%(CHALLENGE))

df = results[['score_mean','score_std']].to_array('metric').stack(stack=['metric','phase']).to_pandas()
df.to_csv(SAVE_DIR+'/challenge_%d.csv'%(CHALLENGE))

ray.shutdown()