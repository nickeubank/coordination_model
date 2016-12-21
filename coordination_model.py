##############
# Coordination game modeled on Siegel 2009
##############
import pandas as pd
import igraph as ig
import numpy.random as npr
import os
import random
import warnings
import sys

try:
    assert sys.version_info >= (3,4)
except: 
    raise EnvironmentError("Not guaranteed to work with less than python 3.4")


def run_coordination_simulation(graph, num_runs=1, num_steps=4, 
                                beta_mean=0.5, beta_std=0.1,
                                debug=False, np_seed=None, 
                                running_test_suite=False):
    """ 
    Master run method. Takes a graph, number of runs,
    number of steps per run, and starting beta mean and std. 
    Returns share in group 1 at end. 

    graph: iGraph object on which to run model
    num_runs: number of times to run simulation
    num_steps: number of steps to execute each run
    beta_mean: mean of normal distribution from which 
               initial preferences (beta) drawn.
    beta_std: std deviation of normal distribution from which 
               initial preferences (beta) drawn.
    debug:  If true, will output graphs of network state at each 
            step of simulation. If `num_runs` more than one, 
            will overright each run, leaving only most recent run.
    np_seed: passed to numpy.random.seed() before beta's drawn. 
    running_test_suite: for internal purposes -- I want tests run
            at function call, but since tests call this 
            function, gets into recursion loop if I don't
            have a flag. 

    """

    # Some basic tests
    if not isinstance(graph, ig.Graph):
        raise TypeError("first argument must by iGraph object!")



    if not running_test_suite:
        print('running test suit')
        test_suite()


    # Start actual run. Create results storage vehicle and start 
    # running simulations!

    results = pd.Series(index=range(num_runs))

    for run in range(num_runs):
        if not running_test_suite:
            print('starting run {}'.format(run))

        results[run] = single_simulation_run(graph, num_steps, beta_mean, 
                                             beta_std, debug, np_seed)

    return results


def single_simulation_run(graph, num_steps, beta_mean, beta_std, debug, np_seed):
    model = Simulation_State(graph, beta_mean, beta_std, debug, np_seed)

    for step in range(num_steps):
        model.iterate(step)

    return model.status.binary_pref.mean()

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
        # For each vertex, look up neighbors. 

        for v in self.graph.vs:
            neighbor_indices = list()
            for n in v.neighbors():
                neighbor_indices.append(n.index)

            # Sends warning if empty slice, but ok. 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.status.loc[v.index, 'local_avg'] = self.status.loc[neighbor_indices, 'binary_pref'].mean()

        # People with no neighbors get 0.5
        self.status.local_avg = self.status.local_avg.fillna(0.5)
        # Check values!

        assert (self.status.local_avg >= 0).all() 
        assert (self.status.local_avg <= 1).all() 

    def plot_graph(self, step='', initial=False, ):
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
                                          beta_mean=1000, beta_std=0.1,
                                          running_test_suite=True) 
    assert len(results) == 10
    assert (results == 1).all()  

    results = run_coordination_simulation(g, num_runs = 10, num_steps=5, 
                                          beta_mean=-1000, beta_std=0.1,
                                          running_test_suite=True) 
    assert len(results) == 10
    assert (results == 0).all()  


    # Make sure deterministic aside from betas. 
    g = ig.Graph.Erdos_Renyi(n=30, p=0.1)
    results = run_coordination_simulation(g, num_runs = 10, num_steps=5, 
                                          beta_mean=0.5, beta_std=0.1,
                                          np_seed=5,
                                          running_test_suite=True) 
    assert (results == results[0]).all()




    print("Test suite successful!")

if __name__ == '__main__':
   pass