import numpy as np
import xarray as xr
import tqdm

from tutlib.util import composition_grid_ternary

def actively_learn(input_dataset,niter,label,extrapolate,choose_next_acquisition,instrument,grid_pts_per_row=100):
    grid = composition_grid_ternary(pts_per_row=grid_pts_per_row,basis=1.0)
    
    
    results = []
    for step in tqdm.tqdm(range(niter)):
        working_dataset = input_dataset.copy()
        working_dataset['a_grid'] = ('grid',grid[:,0])
        working_dataset['b_grid'] = ('grid',grid[:,1])
        working_dataset['c_grid'] = ('grid',grid[:,2])
        working_dataset.attrs['step'] = step
        working_dataset.attrs['components'] = ['c','a','b']
        working_dataset.attrs['components_grid'] = ['c_grid','a_grid','b_grid']
        
        working_dataset = label(working_dataset)
        
        working_dataset = extrapolate(working_dataset)
        
        working_dataset = choose_next_acquisition(working_dataset)
            
        next_sample_dict = working_dataset.attrs['next_sample']
        next_data = instrument.measure(**next_sample_dict)
        
        input_dataset = xr.concat([input_dataset,next_data],dim='sample')
        
        results.append(working_dataset)
    return results