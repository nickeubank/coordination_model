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
    np_seed: passed to numpy.random.seed() before beta's drawn.
            Note the SAME SEED is used on every run -- this is mostly 
            for testing. 
    

    Returns coordination scores between 0 & 1, where 1 means
    everyone coordinated on same candidate, 0 means community split
    50 / 50.

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
                                             debug=debug, 
                                             np_seed=np_seed)
    
        results.loc[run, 'coordination'] = output[0]
        results.loc[run, 'converged'] = output[1]
        results.loc[run, 'steps'] = output[2]


    assert (results.coordination >= 0).all()
    assert (results.coordination <= 1 ).all()

    return results


def single_simulation_run(graph, convergence, num_steps, 
                          convergence_threshold,
                          convergence_period,
                          convergence_max_steps, 
                          beta_mean, beta_std, 
                          debug, np_seed):

    model = Simulation_State(graph, beta_mean, beta_std, debug, np_seed)


    # The model can run in two states -- 
    # Fixed number of steps or convergence. 
    if convergence is not True:
        for step in range(num_steps):
            model.iterate(step)

        converged = False

        # want number of steps in normal numbers, not 0-counting. 
        step +=1 

    if convergence is True:

        continue_running = True
        step = 0
        periods_under_threshold = 0
        converged = False

        while continue_running:
            prior_state = model.status['binary_pref'].copy()

            model.iterate(step)

            post_state = model.status['binary_pref'].copy()

            changes = np.mean(post_state != prior_state)
            assert ((changes >=0) & (changes <=1)).all()

            step +=1

            # Check state!
            if changes < convergence_threshold:
                periods_under_threshold +=1
            
            if periods_under_threshold >= convergence_period:
                continue_running = False
                converged = True

            if step >= convergence_max_steps:
                continue_running = False


    # Now calculate share coordinated in final state. 
    share_for_1 = model.status.binary_pref.mean()

    # Correct so scaled 0 to 1
    coordination = abs(share_for_1 - 0.5) * 2

    return coordination, converged, step

class Simulation_State(object):
    """ 
    General simulation object
    """

    def __init__(self, graph, beta_mean, beta_std, debug, np_seed):
 
        # Copy to protect integrity of original. 
        self.graph = graph.copy()

        # Don't always want to calculate since static.
        self.vcount = graph.vcount()

        self.debug = debug

        # DataFrame to carry all status information. 
        # Row indices will be vertices.
        self.status = pd.DataFrame(columns=['beta','local_avg', 'binary_pref'],
                                   index=range(self.vcount) )

        # Seed betas 
        npr.seed(np_seed)
        self.status['beta'] = npr.normal(loc = beta_mean, scale=beta_std, size=graph.vcount())
        self.status['binary_pref'] = (self.status.beta > 0.5)

        if self.debug:
            self.plot_graph(initial=True)
  
    def iterate(self, step):
        """ 
        Updates model by iterating one step
        """
        self.update_local_avg()
        self.status['binary_pref'] = ((self.status.beta + self.status.local_avg)/2) > 0.5

        if self.debug:
            self.plot_graph(step)

    def update_local_avg(self):

        new_local_avg = pd.Series(index=range(self.vcount), dtype='float')

        # I'm gonna use location indices not names (faster) so want to make sure works. 
        # iGraph uses sequential vertex numbers as ids, so should match. 
        assert (self.status.index == range(self.vcount)).all()

        # For each vertex, look up neighbors. 
        for v in self.graph.vs:
            neighbor_indices = list()
            for n in v.neighbors():
                neighbor_indices.append(n.index)

            # Sends warning if empty slice, but ok. 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_local_avg.iloc[v.index] = self.status.binary_pref.iloc[neighbor_indices].mean()

        self.status.local_avg = new_local_avg

        # People with no neighbors get 0.5
        self.status.local_avg = self.status.local_avg.fillna(0.5)
        # Check values!

        assert (self.status.local_avg >= 0).all() 
        assert (self.status.local_avg <= 1).all() 

    def plot_graph(self, step='', initial=False):
            color_dict = {0:'blue', 1:'red'}

            for v in self.graph.vs:
                new_pref = self.status.binary_pref.loc[v.index]
                v['color'] = color_dict[new_pref]
                v['label'] = 'beta {:.2f},\n local {:.2f}'.format(self.status.beta.loc[v.index], self.status.local_avg.loc[v.index])
        
            debug_folder = '/users/nick/dropbox/GAPP/02_Main Evaluation/Activities' \
                         '/18_voting_and_networks/2_code/coordination_model_folder' \
                         '/debug_plots'

            os.chdir(debug_folder)

            if initial:
                for f in os.listdir("."):
                    if f.endswith(".png"):
                        os.remove(f)

            random.seed('plotseed')
            ig.plot(self.graph, target='plot_{}.png'.format(step), vertex_label_dist=2, margin=70)

            # Don't want affecting anything outside of here, so reset all
            random.seed(None)



def test_suite():

    # All of one type should stay that way
    g = ig.Graph.Erdos_Renyi(n=20, p=0.1)

    results = run_coordination_simulation(g, num_runs = 10, num_steps=5, 
                                          beta_mean=1000, beta_std=0.1) 
    assert len(results.coordination) == 10
    assert (results.coordination == 1).all()  
    assert (results.converged == False).all()  
    assert (results.steps == 5).all()  

    results = run_coordination_simulation(g, num_runs = 10, num_steps=5, 
                                          beta_mean=-1000, beta_std=0.1) 
    assert len(results.coordination) == 10
    assert (results.coordination == 1).all()  


    # Make sure deterministic aside from betas. 
    g = ig.Graph.Erdos_Renyi(n=30, p=0.1)
    results = run_coordination_simulation(g, num_runs = 10, num_steps=5, 
                                          beta_mean=0.5, beta_std=0.1,
                                          np_seed=5) 
    assert (results.coordination == results.coordination[0]).all()

    # Graph with two people should never converge if betas in 0-0.5 and 0.5-1. 
    
    g = ig.Graph([(0,1)])
    
    # Make sure seed generates proper betas. Could vary by operating system? 
    # (I'm on mac)
    npr.seed(1)
    test = npr.normal(0.5, 0.2, 2)
    assert test[0] > 0 and test[0] < 1
    assert test[1] > 0 and test[1] < 1
    
    results = run_coordination_simulation(g, num_runs = 10, num_steps=5, 
                                          beta_mean=0.5, beta_std=0.2,
                                          np_seed=1) 
    assert (results.coordination == 0).all()


    # Now run with convergence threshold. Should never 
    # arrive and should hit it's max iterations. 
    results = run_coordination_simulation(g, num_runs = 10, convergence=True,
                                          convergence_threshold=0.2, 
                                          convergence_period=3,
                                          convergence_max_steps=10,
                                          beta_mean=0.5, beta_std=0.2,
                                          np_seed=1) 

    assert (results.converged == False).all()
    assert (results.steps == 10).all()

    # Change threshold to 1, and should converge nicely (albeit trivially).   
    results = run_coordination_simulation(g, num_runs = 10, convergence=True,
                                          convergence_threshold=1.1, 
                                          convergence_period=3,
                                          convergence_max_steps=10,
                                          beta_mean=0.5, beta_std=0.2,
                                          np_seed=1) 

    assert (results.converged == True).all()
    assert (results.steps == 3).all()



    # Run for immediate convergence
    results = run_coordination_simulation(g, num_runs = 10, convergence=True,
                                          convergence_threshold=0.2, 
                                          convergence_period=3,
                                          convergence_max_steps=10,
                                          beta_mean=100, beta_std=0.1,
                                          np_seed=2) 
    assert (results.converged == True).all()
    assert (results.steps == 3).all()

    # Check threshold values. 
    # Now has two vertices which will flip each time, and 8 that
    # are disconnected and thus static. 
    # So should converge at threshold of 0.21 (since 2 flipping = 0.2)
    # but not at 0.1
    g.add_vertices(range(2,10))
    results = run_coordination_simulation(g, num_runs = 10, convergence=True,
                                          convergence_threshold=0.21, 
                                          convergence_period=3,
                                          convergence_max_steps=10,
                                          beta_mean=0.5, beta_std=0.2,
                                          np_seed=1) 

    assert (results.converged == True).all()
    assert (results.steps == 3).all()


    results = run_coordination_simulation(g, num_runs = 10, convergence=True,
                                          convergence_threshold=0.1, 
                                          convergence_period=3,
                                          convergence_max_steps=10,
                                          beta_mean=0.5, beta_std=0.2,
                                          np_seed=1) 

    assert (results.converged == False).all()
    assert (results.steps == 10).all()




    print("Test suite successful!")

if __name__ == '__main__':
   pass