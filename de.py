#!/usr/bin/env python
import numpy as np
#from pyDOE import lhs
import copy
import random
from scipy.stats import cauchy

class ConvertSolution():
    """
    Repair an infeasible solution on the bbob-mixint. Note that this class is applicable only for the bbob-mixint.

    Attributes
    ----------
    dim: int
        A dimension size
    int_var_2dlist: 2-d integer list
        A list where each element represents an integer value for each dimension. 
    """        
    def __init__(self, dim):
        self.dim = dim
        self.int_var_2dlist = self.bbobmixint_int_list()

    def bbobmixint_int_list(self):
        each_dim = int(self.dim / 5)
        feasible_int_var_2dlist = [
            np.asarray([0, 1]),
            np.asarray([0, 1, 2, 3]),
            np.asarray([0, 1, 2, 3, 4, 5, 6, 7]),
            np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            np.asarray([0])
        ]

        int_var_2dlist = []
        counter = 1
        for i in range(self.dim):
            int_var_2dlist.append(feasible_int_var_2dlist[counter-1])
            if i+1 >= counter * each_dim:
                counter += 1

        return int_var_2dlist    

    def getNearestValue(self, arr, num):
        """
        Copied from https://qiita.com/icchi_h/items/fc0df3abb02b51f81657
        """
        idx = np.abs(arr - num).argmin()
        return arr[idx]

    def round_vec(self, x):
        y = np.copy(x)
        for i in range(self.dim):
            # if |int_var_2dlist[i]| = 1, the last dim - i variables are numerical
            if len(self.int_var_2dlist[i]) == 1:
                break
            else:
                y[i] = self.getNearestValue(self.int_var_2dlist[i], y[i])
        return y

"""
The three test functions were derived from pycma (https://github.com/CMA-ES/pycma)
"""
def sphere(x):
    return sum(np.asarray(x)**2)

def rastrigin(x):
    """Rastrigin test objective function"""
    if not np.isscalar(x[0]):
        N = len(x[0])
        return [10 * N + sum(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x]
    # return 10*N + sum(x**2 - 10*np.cos(2*np.pi*x), axis=1)
    N = len(x)
    return 10 * N + sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rosen(x, alpha=1e2):
    """Rosenbrock test objective function"""
    x = [x] if np.isscalar(x[0]) else x  # scalar into list
    x = np.asarray(x)
    f = [sum(alpha * (x[:-1]**2 - x[1:])**2 + (1. - x[:-1])**2) for x in x]
    return f if len(f) > 1 else f[0]  # 1-element-list into scalar

class XLogger():
    """
    Take a log and show information about the best-so-far solution.

    Attributes
    ----------
    fevals : int
        The number of function evaluations
    max_fevals : int
        The maximum number of function evaluations
    bsf_x: 1-d integer ndarray
        The best-so-far solution
    bsf_fit : float
        The objective value of the best-so-far solution
    runtime_prof_list: dict list
        A list that maintains the number of function evaluations where bsf_x is updated and its objective value
    """      
    def __init__(self, max_fevals, use_logger):
        self.fevals = 0
        self.max_fevals = max_fevals
        self.use_logger = use_logger

        self.bsf_x = None
        self.bsf_obj = np.inf

        if self.use_logger:
            print("#fevals,bsf obj. val.,bsf x")
            
    def update_bsf_x(self, obj_value, x):
        self.fevals += 1
        if self.use_logger and (obj_value < self.bsf_obj):
            self.bsf_obj = obj_value                
            self.bsf_x = np.copy(x)                    
            print("{},{:8e}".format(self.fevals, self.bsf_obj))
            return True
        else:
            return False
            
    def not_happy(self):
        if self.fevals < self.max_fevals:
            return True
        else:
            return False

class Individual:
    """
    An individual in the population

    Attributes
    ----------
    fevals : int
        The number of function evaluations.
    max_fevals : int
        The maximum number of function evaluations.
    x: 1-d integer ndarray
        A solution. 
    fit : float
        The fitness (or objective) value of x
    """  
    def __init__(self, n_var, lbounds, ubounds, init_obj_value=0):
        self.x = lbounds + (ubounds - lbounds) * np.random.rand(n_var)
        self.obj_value = init_obj_value
        self.scale_factor = 0.5
        self.cross_rate = 0.9
        
class DE():
    """
    Collection of functions in DE such as the mutation strategy and crossover
    """
    def __init__(self, fun, dim, lbounds, ubounds, max_fevals, pop_size=100, de_strategy='rand_1', de_sf=0.5, de_cr=0.9, p_best_rate=0.05, archive_rate=1, success_criterion='conventional', int_handling='Baldwin', use_logger=True):
        self.fun = fun
        self.dim = dim
        self.lbounds = lbounds
        self.ubounds = ubounds
        self.max_fevals = max_fevals
        self.pop_size = pop_size
        self.de_strategy = de_strategy
        self.de_sf = de_sf
        self.de_cr = de_cr
        self.p_best_rate = p_best_rate
        self.archive_rate = archive_rate        
        self.p_best_size = max(int(round(self.p_best_rate * self.pop_size)), 2)
        self.archive_size = int(np.floor(self.archive_rate * self.pop_size))
        self.success_criterion = success_criterion
        self.int_handling = int_handling
        self.bbob_round = ConvertSolution(self.dim)        
        self.use_logger = use_logger
        
    # this function is implemented in each derived class
    def run(self):
        pass  
      
    def differential_mutation(self, de_strategy, target_id, best_id, p_best_id, ids, pop, archive, sf=0.5):
        r1, r2, r3, r4, r5 = ids[0], ids[1], ids[2], ids[3], ids[4]
        if de_strategy == 'rand_1':
            v = pop[r1].x + sf * (pop[r2].x - pop[r3].x)
        elif de_strategy == 'rand_2':
            v = pop[r1].x + sf * (pop[r2].x - pop[r3].x) + sf * (pop[r4].x - pop[r5].x)
        elif de_strategy == 'best_1':
            v = pop[best_id].x + sf * (pop[r1].x - pop[r2].x)
        elif de_strategy == 'best_2':
            v = pop[best_id].x + sf * (pop[r1].x - pop[r2].x) + sf * (pop[r3].x - pop[r4].x)
        elif de_strategy == 'current_to_rand_1':
            # This implementation of the current-to-rand/1 strategy is not standard in the DE community.
            v = pop[target_id].x + sf * (pop[r1].x - pop[target_id].x) + sf * (pop[r2].x - pop[r3].x)
            # Traditionally, the scale factor value for the first difference vector is randomly generated in [0,1] as follows:
            #K = np.random.rand()
            #v = pop[target_id] + K * (pop[r1] - pop[target_id]) + sf * (pop[r2] - pop[r3])
        elif de_strategy == 'current_to_best_1':
            v = pop[target_id].x + sf * (pop[best_id].x - pop[target_id].x) + sf * (pop[r1].x - pop[r2].x)
        elif de_strategy == 'current_to_pbest_1':
            if r2 >= self.pop_size:                
                r2 -= self.pop_size
                v = pop[target_id].x + sf * (pop[p_best_id].x - pop[target_id].x) + sf * (pop[r1].x - archive[r2].x)
            else:
                v = pop[target_id].x + sf * (pop[p_best_id].x - pop[target_id].x) + sf * (pop[r1].x - pop[r2].x)
        elif de_strategy == 'rand_to_pbest_1':
            if r3 >= self.pop_size:                
                r3 -= self.pop_size
                v = pop[r1].x + sf * (pop[p_best_id].x - pop[r1].x) + sf * (pop[r2].x - archive[r3].x)
            else:
                v = pop[r1].x + sf * (pop[p_best_id].x - pop[r1].x) + sf * (pop[r2].x - pop[r3].x)
                                            
        # This repair method is used in JADE
        # After the repair, a violated element is in the middle of the target vector and the correspoinding bound
        v = np.where(v < self.lbounds, (self.lbounds + pop[target_id].x) / 2.0, v)
        v = np.where(v > self.ubounds, (self.ubounds + pop[target_id].x) / 2.0, v)

        if self.int_handling == 'Lamarckian':
            v = self.bbob_round.round_vec(v)
        
        return v

    def parent_ids(self, de_strategy, target_id, best_id, p_best_id, arc_size=0):
        # randomly select parent indices such that i != r1 != r2 != ...
        r1 = r2 = r3 = r4 = r5 = target_id
        if de_strategy == 'rand_1' or de_strategy == 'current_to_rand_1':
            while r1 == target_id:
                r1 = np.random.randint(self.pop_size)
            while r2 == target_id or r2 == r1:
                r2 = np.random.randint(self.pop_size)
            while r3 == target_id or r3 == r2 or r3 == r1:
                r3 = np.random.randint(self.pop_size)
        elif de_strategy == 'rand_2':
            while r1 == target_id:
                r1 = np.random.randint(self.pop_size)
            while r2 == target_id or r2 == r1:
                r2 = np.random.randint(self.pop_size)
            while r3 == target_id or r3 == r2 or r3 == r1:
                r3 = np.random.randint(self.pop_size)
            while r4 == target_id or r4 == r3 or r4 == r2 or r4 == r1:
                r4 = np.random.randint(self.pop_size)
            while r5 == target_id or r5 == r4 or r5 == r3 or r5 == r2 or r5 == r1:
                r5 = np.random.randint(self.pop_size)
        elif de_strategy == 'best_1' or de_strategy == 'current_to_best_1':
            while r1 == target_id or r1 == best_id:
                r1 = np.random.randint(self.pop_size)
            while r2 == target_id or r2 == best_id or r2 == r1:
                r2 = np.random.randint(self.pop_size)
        elif de_strategy == 'best_2':
            while r1 == target_id or r1 == best_id:
                r1 = np.random.randint(self.pop_size)
            while r2 == target_id or r2 == best_id or r2 == r1:
                r2 = np.random.randint(self.pop_size)
            while r3 == target_id or r3 == best_id or r3 == r2 or r3 == r1:
                r3 = np.random.randint(self.pop_size)
            while r4 == target_id or r4 == best_id or r4 == r3 or r4 == r2 or r4 == r1:
                r4 = np.random.randint(self.pop_size)
        elif de_strategy == 'current_to_pbest_1':
            # while r1 == target_id or r1 == p_best_id:
            #     r1 = np.random.randint(self.pop_size)
            # while r2 == target_id or r2 == p_best_id or r2 == r1:
            #     r2 = np.random.randint(self.pop_size + arc_size)
            # This implementation allows 
            while r1 == target_id:
                r1 = np.random.randint(self.pop_size)
            while r2 == target_id or r2 == r1:
                r2 = np.random.randint(self.pop_size + arc_size)
        elif de_strategy == 'rand_to_pbest_1':
            # while r1 == target_id or r1 == p_best_id:
            #     r1 = np.random.randint(self.pop_size)
            # while r2 == target_id or r2 == p_best_id or r2 == r1:
            #     r2 = np.random.randint(self.pop_size)
            # while r3 == target_id or r3 == p_best_id or r3 == r2 or r3 == r1:
            #     r3 = np.random.randint(self.pop_size + arc_size)
            while r1 == target_id:
                r1 = np.random.randint(self.pop_size)
            while r2 == target_id or r2 == r1:
                r2 = np.random.randint(self.pop_size)
            while r3 == target_id or r3 == r2 or r3 == r1:
                r3 = np.random.randint(self.pop_size + arc_size)
        else:
            raise Exception('Error: %s is not defined' % de_strategy)
        ids = [r1, r2, r3, r4, r5]
        return ids

    def binomial_crossover(self, x, v, cr=0.9):
        rnd_vals = np.random.rand(self.dim)
        j_rand = np.random.randint(self.dim)
        rnd_vals[j_rand] = 0.0
        u = np.where(rnd_vals <= cr, v, x)
        return u

    def check_success(self, pop, children):
        # Determine if each parameter is (strictly) successful or not.
        is_successful = np.full(self.pop_size, False)
        if self.success_criterion == 'conventional':
            for i in range(self.pop_size):
                if children[i].fit <= pop[i].fit:
                    is_successful[i] = True
        elif self.success_criterion == 'strict':
            for i in range(self.pop_size):
                if children[i].fit < pop[i].fit:
                    is_successful[i] = True    
        return is_successful
    
class SynchronousDE(DE):
    """
    DE with the synchronous population model, i.e., the most basic DE:
    R. Storn and K. Price. Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. J. Glo. Opt., 11(4):341–359, 1997.
    """
    def __init__(self, fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger):
        super().__init__(fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)        
    def run(self):
        x_logger = XLogger(max_fevals=self.max_fevals, use_logger=self.use_logger)
        
        # Initialization of the population
        pop = []
        archive = []        
        for i in range(self.pop_size):
            ind = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
            if self.int_handling == 'Lamarckian':
                ind.x = self.bbob_round.round_vec(ind.x)
            ind.fit = self.fun(ind.x)
            is_updated = x_logger.update_bsf_x(ind.fit, ind.x)            
            if is_updated and __name__ == '__main__':
                if x_logger.bsf_obj < 1e-8:
                    return(x_logger.bsf_obj)
            pop.append(ind)                

        # Each step is repeatedly performed until fevals < max_fevals 
        while x_logger.not_happy():
            children = []
            # Find the best and pbest individuals for best/1, best/2, current-to-pbest/1, rand-to-pbest/1, ...
            fit_arr = np.array([pop[i].fit for i in range(self.pop_size)])
            best_id = np.argmin(fit_arr)
            p_top_ids = np.argsort(fit_arr)[:self.p_best_size]
        
            for i in range(self.pop_size):
                p_best_id = np.random.choice(p_top_ids)
                ids = self.parent_ids(self.de_strategy, i, best_id, p_best_id, len(archive))
                v = self.differential_mutation(self.de_strategy, i, best_id, p_best_id, ids, pop, archive, self.de_sf)
                u = self.binomial_crossover(pop[i].x, v, self.de_cr)
                
                child = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
                child.x = np.copy(u)
                child.fit = self.fun(child.x)

                # Check if the best-sof-far solution is updated or not
                is_updated = x_logger.update_bsf_x(child.fit, child.x)
                if is_updated and __name__ == '__main__':
                    if x_logger.bsf_obj < 1e-8:
                        return(x_logger.bsf_obj)
                    
                children.append(child)

            # Envirionmental selection and the update of the archive
            for i in range(self.pop_size):
                if children[i].fit <= pop[i].fit:
                    archive.append(copy.deepcopy(pop[i]))
                    pop[i] = copy.deepcopy(children[i])

            # Randomly selected individual is removed until |A| < archive_size
            while len(archive) > self.archive_size:
                r = np.random.randint(len(archive))
                del archive[r]

class DERSF(DE):
    def __init__(self, fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger):
        super().__init__(fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)        
    def run(self):
        x_logger = XLogger(max_fevals=self.max_fevals, use_logger=self.use_logger)

        # Initialization of the population        
        pop = []
        archive = []
        for i in range(self.pop_size):
            ind = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
            if self.int_handling == 'Lamarckian':
                ind.x = self.bbob_round.round_vec(ind.x)
            ind.fit = self.fun(ind.x)
            is_updated = x_logger.update_bsf_x(ind.fit, ind.x)            
            if is_updated and __name__ == '__main__':
                if x_logger.bsf_obj < 1e-8:
                    return(x_logger.bsf_obj)
            pop.append(ind)

        # Each step is repeatedly performed until fevals < max_fevals         
        while x_logger.not_happy():
            children = []

            # Generate the two parameters (scale factor and crossover rate) for all individuals in the population
            pop_sf = np.random.uniform(0.5, 1., self.pop_size)
            
            # Find the best and pbest individuals for best/1, best/2, current-to-pbest/1, rand-to-pbest/1, ...            
            fit_arr = np.array([pop[i].fit for i in range(self.pop_size)])
            best_id = np.argmin(fit_arr)
            p_top_ids = np.argsort(fit_arr)[:self.p_best_size]

            # Each individual generates a child            
            for i in range(self.pop_size):
                p_best_id = np.random.choice(p_top_ids)
                ids = self.parent_ids(self.de_strategy, i, best_id, p_best_id, len(archive))
                v = self.differential_mutation(self.de_strategy, i, best_id, p_best_id, ids, pop, archive, pop_sf[i])
                u = self.binomial_crossover(pop[i].x, v, self.de_cr)
                
                child = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
                child.x = np.copy(u)                
                child.fit = self.fun(child.x)

                # Check if the best-sof-far solution is updated or not                
                is_updated = x_logger.update_bsf_x(child.fit, child.x)
                if is_updated and __name__ == '__main__':
                    if x_logger.bsf_obj < 1e-8:
                        return(x_logger.bsf_obj)
                    
                children.append(child)

            # Envirionmental selection and the update of the archive            
            for i in range(self.pop_size):
                if children[i].fit <= pop[i].fit:
                    archive.append(copy.deepcopy(pop[i]))
                    pop[i] = copy.deepcopy(children[i])

            # Randomly selected individual is removed until |A| < archive_size                    
            while len(archive) > self.archive_size:
                r = np.random.randint(len(archive))
                del archive[r]
                
class SinDE(DE):
    def __init__(self, fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger):
        super().__init__(fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)        
    def run(self):
        x_logger = XLogger(max_fevals=self.max_fevals, use_logger=self.use_logger)

        # Initialization of the population        
        pop = []
        archive = []
        for i in range(self.pop_size):
            ind = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
            if self.int_handling == 'Lamarckian':
                ind.x = self.bbob_round.round_vec(ind.x)
            ind.fit = self.fun(ind.x)
            is_updated = x_logger.update_bsf_x(ind.fit, ind.x)            
            if is_updated and __name__ == '__main__':
                if x_logger.bsf_obj < 1e-8:
                    return(x_logger.bsf_obj)
            pop.append(ind)

        # Initialization of parameters for parameter control
        sinde_freq = 0.25
        max_iters = (self.max_fevals / self.pop_size) - 2
        curr_iter = 0
        
        # Each step is repeatedly performed until fevals < max_fevals         
        while x_logger.not_happy():
            children = []

            # Generate the two parameters (scale factor and crossover rate) for all individuals in the population
            sinde_sf = 0.5 * (np.sin(2.0 * np.pi * sinde_freq * curr_iter) * (curr_iter / float(max_iters)) + 1.)
            sinde_cr = 0.5 * (np.sin(2.0 * np.pi * sinde_freq * curr_iter + np.pi) * (curr_iter / float(max_iters)) + 1.)
            
            # Find the best and pbest individuals for best/1, best/2, current-to-pbest/1, rand-to-pbest/1, ...            
            fit_arr = np.array([pop[i].fit for i in range(self.pop_size)])
            best_id = np.argmin(fit_arr)
            p_top_ids = np.argsort(fit_arr)[:self.p_best_size]

            # Each individual generates a child            
            for i in range(self.pop_size):
                p_best_id = np.random.choice(p_top_ids)
                ids = self.parent_ids(self.de_strategy, i, best_id, p_best_id, len(archive))
                v = self.differential_mutation(self.de_strategy, i, best_id, p_best_id, ids, pop, archive, sinde_sf)
                u = self.binomial_crossover(pop[i].x, v, sinde_cr)
                
                child = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
                child.x = np.copy(u)                
                child.fit = self.fun(child.x)

                # Check if the best-sof-far solution is updated or not                
                is_updated = x_logger.update_bsf_x(child.fit, child.x)
                if is_updated and __name__ == '__main__':
                    if x_logger.bsf_obj < 1e-8:
                        return(x_logger.bsf_obj)
                    
                children.append(child)

            # Envirionmental selection and the update of the archive            
            for i in range(self.pop_size):
                if children[i].fit <= pop[i].fit:
                    archive.append(copy.deepcopy(pop[i]))
                    pop[i] = copy.deepcopy(children[i])

            # Randomly selected individual is removed until |A| < archive_size                    
            while len(archive) > self.archive_size:
                r = np.random.randint(len(archive))
                del archive[r]

            curr_iter += 1

class CoDE(DE):
    def __init__(self, fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger):
        super().__init__(fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)        
    def run(self):
        x_logger = XLogger(max_fevals=self.max_fevals, use_logger=self.use_logger)

        # Initialization of the population        
        pop = []
        archive = []
        for i in range(self.pop_size):
            ind = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
            if self.int_handling == 'Lamarckian':
                ind.x = self.bbob_round.round_vec(ind.x)
            ind.fit = self.fun(ind.x)
            is_updated = x_logger.update_bsf_x(ind.fit, ind.x)            
            if is_updated and __name__ == '__main__':
                if x_logger.bsf_obj < 1e-8:
                    return(x_logger.bsf_obj)
            pop.append(ind)

        # Initialization of parameters for parameter control
        code_param_pair_list = [
            [1.0, 0.1],
            [1.0, 0.9],
            [0.8, 0.2]
        ]
        
        # Each step is repeatedly performed until fevals < max_fevals         
        while x_logger.not_happy():
            children = []

            # Generate the two parameters (scale factor and crossover rate) for all individuals in the population
            r_arr = np.random.randint(len(code_param_pair_list), size=self.pop_size)
            pop_sf = [code_param_pair_list[r][0] for r in r_arr]
            pop_cr = [code_param_pair_list[r][1] for r in r_arr]
            
            # Find the best and pbest individuals for best/1, best/2, current-to-pbest/1, rand-to-pbest/1, ...            
            fit_arr = np.array([pop[i].fit for i in range(self.pop_size)])
            best_id = np.argmin(fit_arr)
            p_top_ids = np.argsort(fit_arr)[:self.p_best_size]

            # Each individual generates a child            
            for i in range(self.pop_size):
                p_best_id = np.random.choice(p_top_ids)
                ids = self.parent_ids(self.de_strategy, i, best_id, p_best_id, len(archive))
                v = self.differential_mutation(self.de_strategy, i, best_id, p_best_id, ids, pop, archive, pop_sf[i])
                u = self.binomial_crossover(pop[i].x, v, pop_cr[i])
                
                child = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
                child.x = np.copy(u)                
                child.fit = self.fun(child.x)

                # Check if the best-sof-far solution is updated or not                
                is_updated = x_logger.update_bsf_x(child.fit, child.x)
                if is_updated and __name__ == '__main__':
                    if x_logger.bsf_obj < 1e-8:
                        return(x_logger.bsf_obj)
                    
                children.append(child)

            # Envirionmental selection and the update of the archive            
            for i in range(self.pop_size):
                if children[i].fit <= pop[i].fit:
                    archive.append(copy.deepcopy(pop[i]))
                    pop[i] = copy.deepcopy(children[i])

            # Randomly selected individual is removed until |A| < archive_size                    
            while len(archive) > self.archive_size:
                r = np.random.randint(len(archive))
                del archive[r]
            
class jDE(DE):
    def __init__(self, fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger):
        super().__init__(fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)        
    def run(self):
        x_logger = XLogger(max_fevals=self.max_fevals, use_logger=self.use_logger)

        # Initialization of the population        
        pop = []
        archive = []
        for i in range(self.pop_size):
            ind = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
            if self. int_handling == 'Lamarckian':
                ind.x = self.bbob_round.round_vec(ind.x)
            ind.fit = self.fun(ind.x)
            is_updated = x_logger.update_bsf_x(ind.fit, ind.x)            
            if is_updated and __name__ == '__main__':
                if x_logger.bsf_obj < 1e-8:
                    return(x_logger.bsf_obj)
            pop.append(ind)

        # Initialization of parameters for parameter control
        pop_sf = np.full(self.pop_size, 0.5)
        pop_cr = np.full(self.pop_size, 0.9)
        tau_sf = 0.1
        tau_cr = 0.1

        # Each step is repeatedly performed until fevals < max_fevals         
        while x_logger.not_happy():
            children = []

            # Generate the two parameters (scale factor and crossover rate) for all individuals in the population
            children_sf = np.copy(pop_sf)
            children_cr = np.copy(pop_cr)

            rnd_vals = np.random.rand(self.pop_size)
            children_sf = np.where(rnd_vals <= tau_sf, (1 - 0.1) * np.random.rand(self.pop_size) + 0.1, children_sf)
            children_cr = np.where(rnd_vals <= tau_cr, np.random.rand(self.pop_size), children_cr)            

            # Find the best and pbest individuals for best/1, best/2, current-to-pbest/1, rand-to-pbest/1, ...            
            fit_arr = np.array([pop[i].fit for i in range(self.pop_size)])
            best_id = np.argmin(fit_arr)
            p_top_ids = np.argsort(fit_arr)[:self.p_best_size]

            # Each individual generates a child            
            for i in range(self.pop_size):
                p_best_id = np.random.choice(p_top_ids)
                ids = self.parent_ids(self.de_strategy, i, best_id, p_best_id, len(archive))
                v = self.differential_mutation(self.de_strategy, i, best_id, p_best_id, ids, pop, archive, children_sf[i])
                u = self.binomial_crossover(pop[i].x, v, children_cr[i])
                
                child = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
                child.x = np.copy(u)                
                child.fit = self.fun(child.x)

                # Check if the best-sof-far solution is updated or not                
                is_updated = x_logger.update_bsf_x(child.fit, child.x)
                if is_updated and __name__ == '__main__':
                    if x_logger.bsf_obj < 1e-8:
                        return(x_logger.bsf_obj)
                    
                children.append(child)

            # Determine if each parameter is (strictly) successful or not.
            is_successful = self.check_success(pop, children)

            # Envirionmental selection and the update of the archive            
            for i in range(self.pop_size):
                if children[i].fit <= pop[i].fit:
                    archive.append(copy.deepcopy(pop[i]))
                    pop[i] = copy.deepcopy(children[i])

            # Randomly selected individual is removed until |A| < archive_size                    
            while len(archive) > self.archive_size:
                r = np.random.randint(len(archive))
                del archive[r]

            # Update the parameters
            for i in range(self.pop_size):
                if is_successful[i]:
                    pop_sf[i] = children_sf[i]
                    pop_cr[i] = children_cr[i]
                    
def JADE_sf(loc_param):
    sf = -1.
    while sf <= 0:
        sf = cauchy.rvs(loc=loc_param, scale=0.1)
        sf = min(sf, 1)
    return sf
                                        
class JADE(DE):
    def __init__(self, fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger):
        super().__init__(fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)        
    def run(self):
        x_logger = XLogger(max_fevals=self.max_fevals, use_logger=self.use_logger)

        # Initialization of the population
        pop = []
        archive = []
        for i in range(self.pop_size):
            ind = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
            if self.int_handling == 'Lamarckian':
                ind.x = self.bbob_round.round_vec(ind.x)
            ind.fit = self.fun(ind.x)
            is_updated = x_logger.update_bsf_x(ind.fit, ind.x)            
            if is_updated and __name__ == '__main__':
                if x_logger.bsf_obj < 1e-8:
                    return(x_logger.bsf_obj)
            pop.append(ind)

        # Initialization of parameters for parameter control
        mean_sf = 0.5
        mean_cr = 0.5
        jade_learning_rate = 0.1

        # Each step is repeatedly performed until fevals < max_fevals                 
        while x_logger.not_happy():
            children = []
            
            # Generate the two parameters (scale factor and crossover rate) for all individuals in the population
            children_sf = [JADE_sf(mean_sf) for i in range(self.pop_size)]            
            children_cr = np.random.normal(loc=mean_cr, scale=0.1, size=self.pop_size)
            children_cr = np.clip(children_cr, 0, 1)

            # Find the best and pbest individuals for best/1, best/2, current-to-pbest/1, rand-to-pbest/1, ...            
            fit_arr = np.array([pop[i].fit for i in range(self.pop_size)])
            best_id = np.argmin(fit_arr)
            p_top_ids = np.argsort(fit_arr)[:self.p_best_size]

            # Each individual generates a child            
            for i in range(self.pop_size):
                p_best_id = np.random.choice(p_top_ids)
                ids = self.parent_ids(self.de_strategy, i, best_id, p_best_id, len(archive))
                v = self.differential_mutation(self.de_strategy, i, best_id, p_best_id, ids, pop, archive, children_sf[i])
                u = self.binomial_crossover(pop[i].x, v, children_cr[i])
                
                child = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
                child.x = np.copy(u)
                child.fit = self.fun(child.x)

                # Check if the best-sof-far solution is updated or not                       
                is_updated = x_logger.update_bsf_x(child.fit, child.x)
                if is_updated and __name__ == '__main__':
                    if x_logger.bsf_obj < 1e-8:
                        return(x_logger.bsf_obj)
                    
                children.append(child)

            # Determine if each parameter is (strictly) successful or not.
            is_successful = self.check_success(pop, children)

            # Envirionmental selection and the update of the archive        
            for i in range(self.pop_size):
                if children[i].fit <= pop[i].fit:
                    archive.append(copy.deepcopy(pop[i]))
                    pop[i] = copy.deepcopy(children[i])

            # Randomly selected individual is removed until |A| < archive_size                             
            while len(archive) > self.archive_size:
                r = np.random.randint(len(archive))
                del archive[r]

            # Update the parameters                
            if np.any(is_successful):
                children_sf = np.array(children_sf)
                children_cr = np.array(children_cr)
                succ_sf_arr = children_sf[is_successful]
                succ_cr_arr = children_cr[is_successful]
                                
                mean_sf = (1 - jade_learning_rate) * mean_sf + jade_learning_rate * (np.sum(succ_sf_arr**2) / np.sum(succ_sf_arr))
                mean_cr = (1 - jade_learning_rate) * mean_cr + jade_learning_rate * np.mean(succ_cr_arr)
                
class SHADE(DE):
    def __init__(self, fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger):
        super().__init__(fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)        
    def run(self):
        x_logger = XLogger(max_fevals=self.max_fevals, use_logger=self.use_logger)

        # Initialization of the population        
        pop = []
        archive = []        
        for i in range(self.pop_size):
            ind = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
            if self.int_handling == 'Lamarckian':
                ind.x = self.bbob_round.round_vec(ind.x)
            ind.fit = self.fun(ind.x)
            is_updated = x_logger.update_bsf_x(ind.fit, ind.x)            
            if is_updated and __name__ == '__main__':
                if x_logger.bsf_obj < 1e-8:
                    return(x_logger.bsf_obj)
            pop.append(ind)

        # Initialization of parameters for parameter control            
        shade_memory_size = 10
        memory_sf = np.full(shade_memory_size, 0.5)
        memory_cr = np.full(shade_memory_size, 0.5)        
        memory_pos = 0

        # Each step is repeatedly performed until fevals < max_fevals            
        while x_logger.not_happy():
            children = []

            # Generate the two parameters (scale factor and crossover rate) for all individuals in the population
            rnd_mem_ids = np.random.randint(0, shade_memory_size, self.pop_size)             
            children_sf = [JADE_sf(memory_sf[r]) for r in rnd_mem_ids]
            children_cr = np.random.normal(loc=memory_cr[rnd_mem_ids], scale=0.1, size=self.pop_size)
            children_cr = np.clip(children_cr, 0, 1)

            # Find the best and pbest individuals for best/1, best/2, current-to-pbest/1, rand-to-pbest/1, ...   
            fit_arr = np.array([pop[i].fit for i in range(self.pop_size)])
            best_id = np.argmin(fit_arr)
            p_top_ids = np.argsort(fit_arr)[:self.p_best_size]

            # Each individual generates a child
            for i in range(self.pop_size):
                p_best_id = np.random.choice(p_top_ids)
                ids = self.parent_ids(self.de_strategy, i, best_id, p_best_id, len(archive))
                v = self.differential_mutation(self.de_strategy, i, best_id, p_best_id, ids, pop, archive, children_sf[i])
                u = self.binomial_crossover(pop[i].x, v, children_cr[i])
                
                child = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
                child.x = np.copy(u)
                child.fit = self.fun(child.x)
                
                is_updated = x_logger.update_bsf_x(child.fit, child.x)
                if is_updated and __name__ == '__main__':
                    if x_logger.bsf_obj < 1e-8:
                        return(x_logger.bsf_obj)
                    
                children.append(child)

            # Determine if each parameter is (strictly) successful or not.
            is_successful = self.check_success(pop, children)            

            # Envirionmental selection and the update of the archive            
            for i in range(self.pop_size):
                if children[i].fit <= pop[i].fit:
                    archive.append(copy.deepcopy(pop[i]))
                    pop[i] = copy.deepcopy(children[i])

            # Randomly selected individual is removed until |A| < archive_size
            while len(archive) > self.archive_size:
                r = np.random.randint(len(archive))
                del archive[r]

            # Update the parameters
            if np.any(is_successful):            
                children_sf = np.array(children_sf)
                children_cr = np.array(children_cr)
                succ_sf_arr = children_sf[is_successful]
                succ_cr_arr = children_cr[is_successful]
                memory_sf[memory_pos] = np.sum(succ_sf_arr**2) / np.sum(succ_sf_arr)

                if np.sum(succ_cr_arr) <= 0:
                    memory_cr[memory_pos] = 0
                else:
                    memory_cr[memory_pos] = np.sum(succ_cr_arr**2) / np.sum(succ_cr_arr)
                memory_pos += 1
                memory_pos %= shade_memory_size                

def CoBiDE_sf():
    sf = -1.
    while sf <= 0:
        if np.random.rand() < 0.5:
            sf = cauchy.rvs(loc=0.65, scale=0.1)
        else:
            sf = cauchy.rvs(loc=1., scale=0.1)        
    sf = min(sf, 1)
    return sf

def CoBiDE_cr():
    if np.random.rand() < 0.5:
        cr = cauchy.rvs(loc=0.1, scale=0.1)
    else:
        cr = cauchy.rvs(loc=0.95, scale=0.1)
    cr = np.clip(cr, 0, 1)                          
    return cr
                
class CoBiDE(DE):
    def __init__(self, fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger):
        super().__init__(fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)        
    def run(self):
        x_logger = XLogger(max_fevals=self.max_fevals, use_logger=self.use_logger)

        # Initialization of the population
        pop = []
        archive = []        
        for i in range(self.pop_size):
            ind = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
            if self.int_handling == 'Lamarckian':
                ind.x = self.bbob_round.round_vec(ind.x)
            ind.fit = self.fun(ind.x)
            is_updated = x_logger.update_bsf_x(ind.fit, ind.x)            
            if is_updated and __name__ == '__main__':
                if x_logger.bsf_obj < 1e-8:
                    return(x_logger.bsf_obj)
            pop.append(ind)

        # Initialization of parameters for parameter control            
        pop_sf = np.array([CoBiDE_sf() for i in range(self.pop_size)])
        pop_cr = np.array([CoBiDE_cr() for i in range(self.pop_size)])

        # Each step is repeatedly performed until fevals < max_fevals              
        while x_logger.not_happy():
            children = []
            
            # Find the best and pbest individuals for best/1, best/2, current-to-pbest/1, rand-to-pbest/1, ...            
            fit_arr = np.array([pop[i].fit for i in range(self.pop_size)])
            best_id = np.argmin(fit_arr)
            p_top_ids = np.argsort(fit_arr)[:self.p_best_size]

            # Each individual generates a child
            for i in range(self.pop_size):
                p_best_id = np.random.choice(p_top_ids)
                ids = self.parent_ids(self.de_strategy, i, best_id, p_best_id, len(archive))
                v = self.differential_mutation(self.de_strategy, i, best_id, p_best_id, ids, pop, archive, pop_sf[i])
                u = self.binomial_crossover(pop[i].x, v, pop_cr[i])
                
                child = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
                child.x = np.copy(u)                
                child.fit = self.fun(child.x)

                # Check if the best-sof-far solution is updated or not                         
                is_updated = x_logger.update_bsf_x(child.fit, child.x)
                if is_updated and __name__ == '__main__':
                    if x_logger.bsf_obj < 1e-8:
                        return(x_logger.bsf_obj)
                    
                children.append(child)

            # Determine if each parameter is (strictly) successful or not.
            is_successful = self.check_success(pop, children)

            # Envirionmental selection and the update of the archive                   
            for i in range(self.pop_size):
                if children[i].fit <= pop[i].fit:
                    archive.append(copy.deepcopy(pop[i]))
                    pop[i] = copy.deepcopy(children[i])

            # Randomly selected individual is removed until |A| < archive_size                    
            while len(archive) > self.archive_size:
                r = np.random.randint(len(archive))
                del archive[r]

            # Update the parameters
            new_pop_sf = np.array([CoBiDE_sf() for i in range(self.pop_size)])
            new_pop_cr = np.array([CoBiDE_cr() for i in range(self.pop_size)])            
            pop_sf = np.where(is_successful, pop_sf, new_pop_sf)
            pop_cr = np.where(is_successful, pop_cr, new_pop_cr)

class EPSDE(DE):
    def __init__(self, fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger):
        super().__init__(fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)        
    def run(self):
        x_logger = XLogger(max_fevals=self.max_fevals, use_logger=self.use_logger)

        # Initialization of the population
        pop = []
        archive = []        
        for i in range(self.pop_size):
            ind = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
            if self.int_handling == 'Lamarckian':
                ind.x = self.bbob_round.round_vec(ind.x)
            ind.fit = self.fun(ind.x)
            is_updated = x_logger.update_bsf_x(ind.fit, ind.x)            
            if is_updated and __name__ == '__main__':
                if x_logger.bsf_obj < 1e-8:
                    return(x_logger.bsf_obj)
            pop.append(ind)

        # Initialization of parameters for parameter control
        sf_set = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        cr_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        pop_sf = np.array([np.random.choice(sf_set) for i in range(self.pop_size)])
        pop_cr = np.array([np.random.choice(cr_set) for i in range(self.pop_size)])
                
        # Each step is repeatedly performed until fevals < max_fevals              
        while x_logger.not_happy():
            children = []
            
            # Find the best and pbest individuals for best/1, best/2, current-to-pbest/1, rand-to-pbest/1, ...            
            fit_arr = np.array([pop[i].fit for i in range(self.pop_size)])
            best_id = np.argmin(fit_arr)
            p_top_ids = np.argsort(fit_arr)[:self.p_best_size]

            # Each individual generates a child
            for i in range(self.pop_size):
                p_best_id = np.random.choice(p_top_ids)
                ids = self.parent_ids(self.de_strategy, i, best_id, p_best_id, len(archive))
                v = self.differential_mutation(self.de_strategy, i, best_id, p_best_id, ids, pop, archive, pop_sf[i])
                u = self.binomial_crossover(pop[i].x, v, pop_cr[i])
                
                child = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
                child.x = np.copy(u)                
                child.fit = self.fun(child.x)

                # Check if the best-sof-far solution is updated or not                         
                is_updated = x_logger.update_bsf_x(child.fit, child.x)
                if is_updated and __name__ == '__main__':
                    if x_logger.bsf_obj < 1e-8:
                        return(x_logger.bsf_obj)
                    
                children.append(child)

            # Determine if each parameter is (strictly) successful or not.
            is_successful = self.check_success(pop, children)

            # Envirionmental selection and the update of the archive                   
            for i in range(self.pop_size):
                if children[i].fit <= pop[i].fit:
                    archive.append(copy.deepcopy(pop[i]))
                    pop[i] = copy.deepcopy(children[i])

            # Randomly selected individual is removed until |A| < archive_size                    
            while len(archive) > self.archive_size:
                r = np.random.randint(len(archive))
                del archive[r]

            # Update the parameters
            new_pop_sf = np.array([np.random.choice(sf_set) for i in range(self.pop_size)])
            new_pop_cr = np.array([np.random.choice(cr_set) for i in range(self.pop_size)])
            pop_sf = np.where(is_successful, pop_sf, new_pop_sf)
            pop_cr = np.where(is_successful, pop_cr, new_pop_cr)

class cDE(DE):
    def __init__(self, fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger):
        super().__init__(fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)        
    def run(self):
        x_logger = XLogger(max_fevals=self.max_fevals, use_logger=self.use_logger)

        # Initialization of the population        
        pop = []
        archive = []        
        for i in range(self.pop_size):
            ind = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
            if self.int_handling == 'Lamarckian':
                ind.x = self.bbob_round.round_vec(ind.x)
            ind.fit = self.fun(ind.x)
            is_updated = x_logger.update_bsf_x(ind.fit, ind.x)            
            if is_updated and __name__ == '__main__':
                if x_logger.bsf_obj < 1e-8:
                    return(x_logger.bsf_obj)
            pop.append(ind)

        # Initialization of parameters for parameter control
        cde_param_pair_list = [
            [0.5, 0.0],
            [0.5, 0.5],
            [0.5, 1.0],            
            [0.8, 0.0],
            [0.8, 0.5],
            [0.8, 1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0]
        ]
        cde_n0 = 2.0
        cde_threshold = 1./45        
        n_success_arr = np.zeros(len(cde_param_pair_list))
        
        # Each step is repeatedly performed until fevals < max_fevals            
        while x_logger.not_happy():
            children = []

            # Generate the two parameters (scale factor and crossover rate) for all individuals in the population
            tmp_sum = np.sum(n_success_arr) + len(cde_param_pair_list) * cde_n0
            cde_prob_arr = (n_success_arr + cde_n0) / tmp_sum        
            # Reset the 9 success counters
            if np.any(cde_prob_arr <  cde_threshold):                
                n_success_arr.fill(0.)
                cde_prob_arr.fill(1. / len(cde_param_pair_list))

            param_id_list = np.random.choice(len(cde_param_pair_list), self.pop_size, p=cde_prob_arr)
            pop_sf = [cde_param_pair_list[r][0] for r in param_id_list]
            pop_cr = [cde_param_pair_list[r][1] for r in param_id_list]            
            
            # Find the best and pbest individuals for best/1, best/2, current-to-pbest/1, rand-to-pbest/1, ...   
            fit_arr = np.array([pop[i].fit for i in range(self.pop_size)])
            best_id = np.argmin(fit_arr)
            p_top_ids = np.argsort(fit_arr)[:self.p_best_size]

            # Each individual generates a child
            for i in range(self.pop_size):
                p_best_id = np.random.choice(p_top_ids)
                ids = self.parent_ids(self.de_strategy, i, best_id, p_best_id, len(archive))
                v = self.differential_mutation(self.de_strategy, i, best_id, p_best_id, ids, pop, archive, pop_sf[i])
                u = self.binomial_crossover(pop[i].x, v, pop_cr[i])
                
                child = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
                child.x = np.copy(u)
                child.fit = self.fun(child.x)
                
                is_updated = x_logger.update_bsf_x(child.fit, child.x)
                if is_updated and __name__ == '__main__':
                    if x_logger.bsf_obj < 1e-8:
                        return(x_logger.bsf_obj)
                    
                children.append(child)

            # Determine if each parameter is (strictly) successful or not.
            is_successful = self.check_success(pop, children)            

            # Envirionmental selection and the update of the archive            
            for i in range(self.pop_size):
                if children[i].fit <= pop[i].fit:
                    archive.append(copy.deepcopy(pop[i]))
                    pop[i] = copy.deepcopy(children[i])

            # Randomly selected individual is removed until |A| < archive_size
            while len(archive) > self.archive_size:
                r = np.random.randint(len(archive))
                del archive[r]

            # Update the parameters
            if np.any(is_successful):
                for pid in param_id_list[is_successful]:
                    n_success_arr[pid] += 1

class DECaRS(DE):
    def __init__(self, fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger):
        super().__init__(fun, dim, lbounds, ubounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)        
    def run(self):
        x_logger = XLogger(max_fevals=self.max_fevals, use_logger=self.use_logger)

        # Initialization of the population
        pop = []
        archive = []        
        for i in range(self.pop_size):
            ind = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
            if self.int_handling == 'Lamarckian':
                ind.x = self.bbob_round.round_vec(ind.x)
            ind.fit = self.fun(ind.x)
            is_updated = x_logger.update_bsf_x(ind.fit, ind.x)            
            if is_updated and __name__ == '__main__':
                if x_logger.bsf_obj < 1e-8:
                    return(x_logger.bsf_obj)
            pop.append(ind)

        # Initialization of parameters for parameter control
        sf_min = 0.5
        sf_max = 0.55
        cr_set = [0.5, 0.6, 0.7, 0.8, 0.9]
                
        # Each step is repeatedly performed until fevals < max_fevals              
        while x_logger.not_happy():
            children = []
            
            # Find the best and pbest individuals for best/1, best/2, current-to-pbest/1, rand-to-pbest/1, ...            
            fit_arr = np.array([pop[i].fit for i in range(self.pop_size)])
            best_id = np.argmin(fit_arr)
            p_top_ids = np.argsort(fit_arr)[:self.p_best_size]

            # Generate the crossover rate value for all individuals
            common_cr = np.random.choice(cr_set)                   
            pop_sf = np.random.uniform(sf_min, sf_max, self.pop_size)
            
            # Each individual generates a child
            for i in range(self.pop_size):
                p_best_id = np.random.choice(p_top_ids)
                ids = self.parent_ids(self.de_strategy, i, best_id, p_best_id, len(archive))
                v = self.differential_mutation(self.de_strategy, i, best_id, p_best_id, ids, pop, archive, pop_sf[i])
                u = self.binomial_crossover(pop[i].x, v, common_cr)
                
                child = Individual(n_var=self.dim, lbounds=self.lbounds, ubounds=self.ubounds)
                child.x = np.copy(u)                
                child.fit = self.fun(child.x)

                # Check if the best-sof-far solution is updated or not                         
                is_updated = x_logger.update_bsf_x(child.fit, child.x)
                if is_updated and __name__ == '__main__':
                    if x_logger.bsf_obj < 1e-8:
                        return(x_logger.bsf_obj)
                    
                children.append(child)

            # Envirionmental selection and the update of the archive                   
            for i in range(self.pop_size):
                if children[i].fit <= pop[i].fit:
                    archive.append(copy.deepcopy(pop[i]))
                    pop[i] = copy.deepcopy(children[i])

            # Randomly selected individual is removed until |A| < archive_size                    
            while len(archive) > self.archive_size:
                r = np.random.randint(len(archive))
                del archive[r]
                    
if __name__ == '__main__':
    np.random.seed(seed=1)
    random.seed(1)
    
    fun = sphere
    dim = 10
    lower_bounds = np.full(dim, -5)
    upper_bounds = np.full(dim, 5)    
    max_fevals = 10000 * dim
    
    #de_alg = 'syn_de'
    #de_alg = 'dersf'
    de_alg = 'jde'
    #de_alg = 'jade'
    #de_alg = 'shade'
    #de_alg = 'cobide'
    #de_alg = 'epsde'
    #de_alg = 'dersf'
    #de_alg = 'sinde'
    #de_alg = 'code'
    #de_alg = 'cde'
    
    pop_size = 100
    de_sf = 0.5
    de_cr = 0.9
    # 'rand_1', 'rand_2', 'best_1', 'best_2', 'current_to_best_1', 'current_to_pbest_1', 'rand_to_pbest_1'
    de_strategy = 'rand_2'

    # these are activated only when using 'current_to_pbest_1' and 'rand_to_pbest_1'
    p_best_rate = 0.05
    archive_rate = 1.0

    success_criterion = 'conventional'

    #'Baldwin'
    #Lamarckian
    int_handling = 'Baldwin' #'Lamarckian'
    #int_handling = 'Lamarckian'
    
    use_logger = True
    
    de = None    
    if de_alg == 'syn_de':        
        de = SynchronousDE(fun, dim, lower_bounds, upper_bounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
    elif de_alg == 'jde':        
        de = jDE(fun, dim, lower_bounds, upper_bounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)        
    elif de_alg == 'jade':        
        de = JADE(fun, dim, lower_bounds, upper_bounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)        
    elif de_alg == 'shade':        
        de = SHADE(fun, dim, lower_bounds, upper_bounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
    elif de_alg == 'cobide':        
        de = CoBiDE(fun, dim, lower_bounds, upper_bounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
    elif de_alg == 'epsde':        
        de = EPSDE(fun, dim, lower_bounds, upper_bounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
    elif de_alg == 'dersf':        
        de = DERSF(fun, dim, lower_bounds, upper_bounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
    elif de_alg == 'sinde':        
        de = SinDE(fun, dim, lower_bounds, upper_bounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
    elif de_alg == 'code':        
        de = CoDE(fun, dim, lower_bounds, upper_bounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
    elif de_alg == 'cde':        
        de = cDE(fun, dim, lower_bounds, upper_bounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
    elif de_alg == 'decars':        
        de = DECaRS(fun, dim, lower_bounds, upper_bounds, max_fevals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
    else:
        raise Exception('Error: %s is not defined' % de_alg)
    de.run()    
    
