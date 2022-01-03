from main_const import calc
import numpy as np
import matplotlib.pyplot as plt
from artap.problem import Problem
from artap.algorithm_pymoo import Pymoo
from artap.results import Results
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

class CoolOpt(Problem):

    def set(self):

        # Not mandatory to give a name for the test problem
        self.name = 'CoolOpt'

        self.parameters = [{'name':'nk', 'bounds': [20, 60], 'initial_value': 40, 'parameter_type':'integer'},
                           {'name':'dk', 'bounds': [0.01, 0.03], 'initial_value': 0.025, 'parameter_type':'float'},
                           {'name':'Ds', 'bounds': [0.297, 0.4], 'initial_value': 0.3, 'parameter_type':'float'}]

        self.costs = [{'name': 'Tav', 'criteria': 'minimize'},
                      {'name': 'S_stator', 'criteria': 'maximize'}]
        
        self.constraints = [{'name': 'dP'},
                            {'name': 'const_gap'},
                            {'name': 'const_D'}]        

    def evaluate(self, individuals):
        individuals.vector[0] = round(individuals.vector[0])
        res = calc(individuals.vector[0],individuals.vector[1],individuals.vector[2])
        Tav = res[1]
        S_stator = res[2]
        return [Tav, S_stator]
    
    def evaluate_inequality_constraints(self, x):
        nk = x[0]
        dk = x[1]
        Ds = x[2]
        
        #  Ventilation constrains
        res = calc(nk, dk, Ds)
        dP = res[0]

        #  Geometric constrains
        D1 = 0.297
        D2 = 0.4
        duct_gap = 0.005
        D1_gap = 0.003
        D2_gap = 0.003
        
        hj = (D2 - D1) / 2
    
        const_gap = nk * (dk + duct_gap) / np.pi
        const_D = dk + D1_gap + D2_gap

        inequality_constraints = [dP-1400, const_gap - Ds, const_D - hj]

        return inequality_constraints

# Initialization of the problem
problem = CoolOpt()

moo_algorithm_nsga2 = NSGA2(
    pop_size=100,
    n_offsprings=100,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
)

algorithm = Pymoo(problem)

algorithm.options['verbose_level'] = 0
algorithm.options['n_iterations'] = 10
algorithm.options['max_processes'] = 1
algorithm.options['algorithm'] = moo_algorithm_nsga2
algorithm.run()

res = Results(problem)

solution = res.find_optimum()
Tav = solution.costs[0]
S_stator = solution.costs[1]
nk = solution.vector[0]
dk = solution.vector[1]
Ds = solution.vector[2]
opt_res = {"nk":nk, "dk":dk, "Ds":Ds, "Tav":Tav, "S_stator":S_stator}
print(opt_res)

# Data visualisation

fp_1 = []
fp_2 = []
for individual in problem.last_population():
    fp_1.append(individual.costs[0])
    fp_2.append(individual.costs[1])
    
f_1 = []
f_2 = []
for population in problem.populations().values():
    for individual in population:
        f_1.append(individual.costs[0])
        f_2.append(individual.costs[1])
        

fig, ax = plt.subplots()
ax.scatter(f_2, f_1)
ax.scatter(fp_2, fp_1)


ax.set(xlabel='Solid surface (m2)', ylabel='Temperature (°C)',
        title='Pareto front and optimal solution')
ax.grid()
ax.legend(["Pareto front","Optimal solution"])
