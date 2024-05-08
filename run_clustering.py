from autophasemap import multi_kmeans_run, BaseDataSet
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import sys
import ray
ray.init()

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
  
CHALLENGE = int(sys.argv[1])
num_phases = [2,4,2]
spline_s_values = [1.0, 0.05, 0.025]
result = xr.load_dataset("./figures/challenge_%d_copy/challenge_%d.nc"%(CHALLENGE, CHALLENGE))
C = result[['c', 'a', 'b']].to_array().values.T
data = DataSet(C, result["q"].values, result['sas'].values)
data.generate(spline_s = spline_s_values[CHALLENGE-1])

out,_ = multi_kmeans_run(1, 
                    data, 
                    num_phases[CHALLENGE-1], 
                    max_iter=50, 
                    verbose=3, 
                    grid_dim= 10,
                    smoothen=False
                    )

def plot(data, out):
    n_clusters = len(out.templates)
    fig, axs = plt.subplots(1,n_clusters, figsize=(4*(n_clusters), 4))
    axs = axs.flatten() 
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