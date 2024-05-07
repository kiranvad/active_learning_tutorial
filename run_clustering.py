from autophasemap import multi_kmeans_run, BaseDataSet
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

import ray
ray.init()

num_nodes = len(ray.nodes())
print('Total number of nodes are {}'.format(num_nodes))
print('Avaiable CPUS : ', ray.available_resources()['CPU'])

class DataSet:
  def __init__(self, C, q, Iq):
    self.flags = (Iq>0.01).all(axis=0)
    self.q = q[self.flags]
    self.Iq = Iq[:,self.flags]
    self.C = C
    self.N = self.C.shape[0]
    self.q_log = np.log10(self.q)
    self.n_domain = 300
    self.q_logspace = np.geomspace(self.q_log.min(), self.q_log.max(), self.n_domain)
    self.t = (self.q_logspace-min(self.q_logspace))/(max(self.q_logspace)-min(self.q_logspace))

  def generate(self, spline_w = 1e-1):
    self.F = []
    for i in range(self.N):
      Iq = np.log10(self.Iq[i,:])
      spline = interp.splrep(self.q_log, Iq,
                             w = (1/spline_w)*(np.ones_like(self.q_log)),
                             )
      Iq = interp.splev(self.q_logspace, spline)
      norm = np.sqrt(np.trapz(Iq**2, self.q_logspace))
      self.F.append(Iq)
    
    return 
  
CHALLENGE = 1
num_phases = [2,4,2]
spline_w_values = [1e-2, 5e-3, 5e-2]
result = xr.load_dataset("./figures/challenge_%d_copy/challenge_%d.nc"%(CHALLENGE, CHALLENGE))
C = result[['c', 'a', 'b']].to_array().values.T
data = DataSet(C, result["q"].values, result['sas'].values)
data.generate(spline_w = spline_w_values[CHALLENGE-1])

out,_ = multi_kmeans_run(1, 
                    data, 
                    num_phases[CHALLENGE-1], 
                    max_iter=50, 
                    verbose=3, 
                    grid_dim= 10,
                    smoothen=False
                    )

def tern2cart(T):
    # convert ternary data to cartesian coordinates
    cosd  = lambda d : np.cos(d * np.pi/180)
    sind = lambda d : np.sin(d * np.pi/180)
    sT = np.sum(T,axis = 1)
    T = 100 * T / np.tile(sT[:,None],(1,3))

    C = np.zeros((T.shape[0],2))
    C[:,1] = T[:,1]*sind(60)/100
    C[:,0] = T[:,0]/100 + C[:,1]*sind(30)/sind(60)

    return C

comp_grid = result[['c_grid', 'a_grid', 'b_grid']].to_array().values.T
XYc_sample = tern2cart(data.C)
XYc_grid = tern2cart(comp_grid)

def plot(data, out):
    n_clusters = len(out.templates)
    fig, axs = plt.subplots(1,n_clusters+1, figsize=(4*(n_clusters+1), 4))
    axs = axs.flatten() 
    axs[n_clusters].scatter(XYc_grid[:,0], XYc_grid[:,1], c = result["labels_grid"].to_numpy(), alpha=0.1)
    axs[n_clusters].scatter(XYc_sample[:,0], XYc_sample[:,1], c = out.delta_n, s=50)
    axs[n_clusters].axis('off')
    for k in range(n_clusters):
        Mk = np.argwhere(out.delta_n==k).squeeze()
        for i in Mk:
            axs[k].plot(data.t, data.F[i], color='grey')
        
        axs[k].plot(data.t, out.templates[k], lw=3.0, color='tab:red') 
        axs[k].axis('off')

    return fig, axs

# plot phase map and corresponding spectra
fig, axs = plot(data, out)
plt.savefig('./figures/phase_map_%d.png'%CHALLENGE)
plt.close()

ray.shutdown()