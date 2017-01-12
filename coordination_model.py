##############
# Coordination game modeled on Siegel 2009
##############
import pandas as pd
import igraph as ig
import numpy.random as npr
import numpy as np
import os
import random
import warnings
import sys
from numba import jit
try:
    assert sys.version_info >= (3,4)
except: 
    raise EnvironmentError("Not guaranteed to work with less than python 3.4")


def run_coordination_simulation(graph, num_runs=1, convergence=False,
                                num_steps=None, convergence_threshold=None,
                                convergence_period=None,
                                convergence_max_steps=None,
                                beta_mean=0.5, beta_std=0.1,
                                debug=False, np_seed=None):
    """ 
    Master run method. Takes a graph, number of runs,
    number of steps per run, and starting beta mean and std. 
    Returns share in group 1 at end. 

    graph: iGraph object on which to run model
    num_runs: number of times to run simulation
    convergence: True/False -- whether to use convergence 
              threshold instead of finite number of steps. 
              If True, must specify `convergence_threshold`.
              If False, must specify `num_steps`.
    num_steps: number of steps to execute each run for 
               finite run. Only for use when 
               `convergence=False`.
    convergence_threshold: maximum share of nodes allowed to 
                be changing step to step in converged state. 
    convergence_period: number of steps must stay below
                threshold to be considered converged. 
    convergence_max_steps: limit to steps allowed if converging.  
    beta_mean: mean of normal distribution from which 
               initial preferences (beta) drawn.
    beta_std: std deviation of normal distribution from which 
               initial preferences (beta) drawn.
    debug:  If true, will output graphs of network state at each 
            step of simulation and will run test suite. If 
            `num_runs` more than one, will overwrite each run, 
            leaving only most recent run.
    np_seed: passed to numpy.random.seed().
    

    Returns DataFrame containing, for each run:
    	- `coordination` score: (1 = all same state, 0 = 50-50 in converged state.)
    	- `converged`: did it successfully converge? np.nan if convergence False 
    	               since not actually known if in converged state.
    	- `steps`: number of UNCONVERGED steps. Does NOT include steps during which
    			   network holds in converged state to satisfy 
    			   for `convergence_period` criterion. 

    """

    # Some basic tests
    if not isinstance(graph, ig.Graph):
        raise TypeError("first argument must by iGraph object!")

    if debug:
        print('running test suit')
        test_suite()

    if not convergence:
        if convergence_threshold is not None or convergence_period is not None or convergence_max_steps is not None:
                raise ValueError("Can't use convergence arguments if convergence not true")

    if convergence:
        if num_steps is not None:
            raise ValueError("cannot use num_steps with convergence")
        if convergence_threshold is None or convergence_period is None or convergence_max_steps is None:
            raise ValueError("If convergence is True, must specify all convergence arguments")


    # Set seed.
    npr.seed(np_seed)

    # Start actual run. Create results storage vehicle and start 
    # running simulations!

    results = pd.DataFrame(columns=['coordination', 'converged', 'steps'],
                           index=range(num_runs))


    for run in range(num_runs):
        if debug:
            print('starting run {}'.format(run))

        output = single_simulation_run(graph, convergence=convergence,
                                             num_steps=num_steps, 
                                             convergence_threshold=convergence_threshold,
                                             convergence_period=convergence_period,
                                             convergence_max_steps=convergence_max_steps, 
                                             beta_mean=beta_mean, 
                                             beta_std=beta_std, 
                                             debug=debug)
    
        results.loc[run, 'coordination'] = output[0]
        results.loc[run, 'converged'] = output[1]
        results.loc[run, 'steps'] = output[2]


    # type tweak -- still problems with DataFrame constructor -- can't set different columns to different types.different
    results.coordination = results.coordination.astype('float')
    results.converged = results.converged.astype('bool')
    results.steps = results.steps.astype('float')


    # If run for finite number of steps, don't know if
    # converged. 
    if not convergence:
    	results.converged = np.nan

    # Want steps PRE convergence, so subtract number
    # of steps we require simulation to hold at 
    # converged state. 
    results.loc[results.converged==True, 'steps'] = results.loc[results.converged==True, 'steps'] - convergence_period


    assert (results.coordination >= 0).all()
    assert (results.coordination <= 1 ).all()

    return results


def single_simulation_run(graph, convergence, num_steps, 
                          convergence_threshold,
                          convergence_period,
                          convergence_max_steps, 
                          beta_mean, beta_std, 
                          debug):

    model = Simulation_State(graph, beta_mean, beta_std, debug)


    # The model can run in two states -- 
    # Fixed number of steps or convergence. 
    if convergence is not True:
        for step in range(num_steps):

            model.iterate(step)

        converged = False

        # want number of steps in normal numbers, not 0-counting. 
        step +=1 

    else:

        continue_running = True
        step = 0
        periods_under_threshold = 0
        converged = False

        while continue_running:
            prior_state = model.binary_pref.copy()

            model.iterate(step)

            post_state = model.binary_pref.copy()

            changes = np.mean(post_state != prior_state)
            assert ((changes >=0) & (changes <=1)).all()

            step +=1

            # Check state! If convergence level, count towards convergence period. 
            if changes < convergence_threshold:
                periods_under_threshold +=1
            
            # If did move too much, reset periods_under_threshold counter. 
            if changes >= convergence_threshold:
            	periods_under_threshold = 0

            if periods_under_threshold >= convergence_period:
                continue_running = False
                converged = True

            if step >= convergence_max_steps:
                continue_running = False


    # Now calculate share coordinated in final state. 
    share_for_1 = model.binary_pref.mean()

    # Correct so scaled 0 to 1
    coordination = abs(share_for_1 - 0.5) * 2

    return coordination, converged, step

class Simulation_State(object):
    """ 
    General simulation object
    """

    def __init__(self, graph, beta_mean, beta_std, debug):
 
        # Copy to protect integrity of original. 
        self.graph = graph.copy()

        # Don't always want to calculate since static.
        self.vcount = graph.vcount()

        self.debug = debug

        # Seed status Series.  
        self.beta = pd.Series(npr.normal(loc = beta_mean, scale=beta_std, size=graph.vcount()), 
                              index=range(self.vcount))

        self.binary_pref = (self.beta > 0.5)
        self.local_avg = pd.Series(index=range(self.vcount))

        if self.debug:
            self.plot_graph(initial=True)

    def iterate(self, step):
        """ 
        Updates model by iterating one step.
        Note this has to be done in a random, sequential manner.

        Otherwise, I get "flashing" problems (consider a graph of two nodes with 
        opposite states -- for many values, each will move to other person's state
        each iteration, leading to flipping.)
        """

        for v in npr.permutation(self.vcount):
            self.update_local_avg(v)
            self.binary_pref.iloc[v] = ((self.beta.iloc[v] + self.local_avg.iloc[v])/2) > 0.5

        assert pd.notnull(self.local_avg).all()

        if self.debug:
            self.plot_graph(step)

    def update_local_avg(self, v):

        neighbor_indices = [n.index for n in self.graph.vs[v].neighbors()]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*Mean of empty slice.*")
            warnings.filterwarnings("ignore", ".*invalid value encountered in double_scalars.*")
            a = self.binary_pref.iloc[neighbor_indices].mean()
    
        self.local_avg.iloc[v] = a if pd.notnull(a) else 0.5


    def plot_graph(self, step='', initial=False):
            color_dict = {0:'blue', 1:'red'}

            for v in self.graph.vs:
                new_pref = self.binary_pref.loc[v.index]
                v['color'] = color_dict[new_pref]
                v['label'] = 'beta {:.2f},\n local {:.2f}'.format(self.beta.loc[v.index], self.local_avg.loc[v.index])
        
            debug_folder = '/users/nick/dropbox/GAPP/02_Main Evaluation/Activities' \
                         '/18_voting_and_networks/2_code/coordination_model_folder' \
                         '/debug_plots'

            os.chdir(debug_folder)

            if initial:
                for f in os.listdir("."):
                    if f.endswith(".png"):
                        os.remove(f)

            random.seed('plotseed')
            ig.plot(self.graph, target='plot_{}.png'.format(step), vertex_label_dist=2, 
                    vertex_size=20, vertex_label_size=20, margin=70, bbox=(2000,2000))

            # Don't want affecting anything outside of here, so reset all
            random.seed(None)



def test_suite():

    # All of one type should stay that way
    g = ig.Graph.Erdos_Renyi(n=20, p=0.1)

    results = run_coordination_simulation(g, num_runs = 10, num_steps=5, 
                                          beta_mean=1000, beta_std=0.1) 
    assert len(results.coordination) == 10
    assert (results.coordination == 1).all()  
    assert (pd.isnull(results.converged)).all()  
    assert (results.steps == 5).all()  

    results = run_coordination_simulation(g, num_runs = 10, num_steps=5, 
                                          beta_mean=-1000, beta_std=0.1) 
    assert len(results.coordination) == 10
    assert (results.coordination == 1).all()  


    # Make sure seed works. 
    g = ig.Graph.Erdos_Renyi(n=30, p=0.1)
    run1 = run_coordination_simulation(g, num_runs = 10, num_steps=5, 
                                          beta_mean=0.5, beta_std=0.1,
                                          np_seed=5)

    run2 = run_coordination_simulation(g, num_runs = 10, num_steps=5, 
                                          beta_mean=0.5, beta_std=0.1,
                                          np_seed=5) 
    assert (run1.coordination == run2.coordination).all()


    # Change threshold to 1, and should converge nicely (albeit trivially).   
    results = run_coordination_simulation(g, num_runs = 10, convergence=True,
                                          convergence_threshold=1.1, 
                                          convergence_period=3,
                                          convergence_max_steps=10,
                                          beta_mean=0.5, beta_std=0.2) 
    assert (results.converged == True).all()
    assert (results.steps == 0).all()


    # Change threshold to 0, and should hit threshold
    results = run_coordination_simulation(g, num_runs = 10, convergence=True,
                                          convergence_threshold=0, 
                                          convergence_period=3,
                                          convergence_max_steps=10,
                                          beta_mean=0.5, beta_std=0.2) 

    assert (results.converged == False).all()
    assert (results.steps == 10).all()



    # Set some
    g2 = ig.Graph()
    # Four nodes, first three connected, fourth alone.. 
    g2.add_vertices([0,1,2,3])
    g2.add_edges([(0,1), (1,2), (0,2)])
    npr.seed(44)
    test = npr.normal(loc = 0.5, scale=0.1, size=4)
    # Make sure seed works. May vary with operating system. I'm on a mac
    assert (abs(test - np.array([ 0.42493853,  0.63163573,  0.624614  ,  0.33950843])) < 0.01).all()

    # Threshold to exclude first step. 
    results = run_coordination_simulation(g2, num_runs = 1, convergence=True,
                                          convergence_threshold=0.01, 
                                          convergence_period=3,
                                          convergence_max_steps=10,
                                          beta_mean=0.5, beta_std=0.1,
                                          np_seed=44) 
    assert (results.coordination == 0.5).all()
    assert (results.converged == True).all()
    # In first step, moves from 0, 1, 1, 0 to 1,1,1,0. Then three periods stable. 
    assert (results.steps == 1).all()

    # Higher threshold counts first adjustment as within convergence threshold.
    results = run_coordination_simulation(g2, num_runs = 1, convergence=True,
                                          convergence_threshold=0.3, 
                                          convergence_period=3,
                                          convergence_max_steps=10,
                                          beta_mean=0.5, beta_std=0.1,
                                          np_seed=44) 
    assert (results.coordination == 0.5).all()
    assert (results.converged == True).all()
    # In first step, moves from 0, 1, 1, 0 to 1,1,1,0. STill 3/4 stable, so counts towards
    # convergence. Only three steps now. 
    assert (results.steps == 0).all()







    print("Test suite successful!")

if __name__ == '__main__':
   pass