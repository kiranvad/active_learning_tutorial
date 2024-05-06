from autophasemap import multi_kmeans_run, BaseDataSet
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import ray
ray.init(num_cpus=20)

num_nodes = len(ray.nodes())
print('Total number of nodes are {}'.format(num_nodes))
print('Avaiable CPUS : ', ray.available_resources()['CPU'])

class DataSet(BaseDataSet):
  def __init__(self, C, q, Iq):
    self.flags = (Iq>0.01).all(axis=0)
    self.q = q[self.flags]
    self.Iq = Iq[:,self.flags]
    self.C = C
    self.N = self.C.shape[0]
    self.q_log = np.log10(self.q)
    self.t = (self.q_log-min(self.q_log))/(max(self.q_log)-min(self.q_log))
    super().__init__(n_domain=len(self.q))

    assert self.N==self.Iq.shape[0], "C and Iq should have same number of rows"
    assert self.n_domain==self.Iq.shape[1], "Length of q should match with columns size of Iq"

  def generate(self, window_length = 51):
    self.F = []
    for i in range(self.N):
      f_smooth = savgol_filter(np.log10(self.Iq[i,:]), window_length, 3)
      norm = np.sqrt(np.trapz(f_smooth**2, np.log10(self.q)))
      self.F.append(f_smooth/norm)
    
    return 

CHALLENGE = 3
num_phases = [2,4,2]
window_length = [30, 15, 15]
result = xr.load_dataset("./figures/challenge_%d/challenge_%d.nc"%(CHALLENGE, CHALLENGE))
C = result[['a','b','c']].to_array().values.T
data = DataSet(C, result["q"].values, result['sas'].values)
data.generate(window_length=window_length[CHALLENGE-1])

out = multi_kmeans_run(1, data, num_phases[CHALLENGE-1], max_iter=50, verbose=3, smoothen=False)

# plot phase map and corresponding spectra
def cosd(deg):
    # cosine with argument in degrees
    return np.cos(deg * np.pi/180)

def sind(deg):
    # sine with argument in degrees
    return np.sin(deg * np.pi/180)

def tern2cart(T):
    # convert ternary data to cartesian coordinates
    sT = np.sum(T,axis = 1)
    T = 100 * T / np.tile(sT[:,None],(1,3))

    C = np.zeros((T.shape[0],2))
    C[:,1] = T[:,1]*sind(60)/100
    C[:,0] = T[:,0]/100 + C[:,1]*sind(30)/sind(60)
    return C

XYc = tern2cart(data.C[:,[1,2,0]])

def plot(data, out):
    n_clusters = len(out.templates)
    fig, axs = plt.subplots(1,n_clusters+1, figsize=(4*(n_clusters+1), 4))
    axs = axs.flatten() 
    axs[n_clusters].scatter(XYc[l:,0], XYc[:,1], color = out.delta_n)
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