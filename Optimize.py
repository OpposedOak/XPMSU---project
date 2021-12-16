from main import calc
import matplotlib.pyplot as plt
from artap.problem import Problem
from artap.algorithm_genetic import NSGAII
from artap.results import Results


class CoolOpt(Problem):

    def set(self):

        # Not mandatory to give a name for the test problem
        self.name = 'CoolOpt'

        self.parameters = [{'name':'nk', 'bounds': [20, 60], 'parameter_type':'integer'},
                           {'name':'dk', 'bounds': [0.018, 0.03], 'parameter_type':'float'}]

        self.costs = [{'name': 'dP', 'criteria': 'minimize'},
                      {'name': 'Tav', 'criteria': 'minimize'}]

    def evaluate(self, individuals):
        individuals.vector[0] = round(individuals.vector[0])
        res = calc(individuals.vector[0],individuals.vector[1])
        dP = res[0]
        Tav = res[1]
        return [dP, Tav]

# Initialization of the problem
problem = CoolOpt()

# Perform the optimization iterating over 10 times on 10 individuals.
algorithm = NSGAII(problem)
algorithm.options['max_population_number'] = 100
algorithm.options['max_population_size'] = 100
algorithm.run()

# Post - processing the results
# reads in the result values into the res results class
res = Results(problem)
# finding the optimal
solution = res.find_optimum()
pareto = res.pareto_front()
# pareto = res.pareto_values()


dP = solution.costs[0]
Tav = solution.costs[1]
nk = solution.vector[0]
dk = solution.vector[1]

opt_res = {"nk":nk, "dk":dk, "dP":dP, "Tav":Tav}
print(opt_res)

# Data visualisation

fig, ax = plt.subplots()
ax.scatter(pareto[0],pareto[1])
ax.scatter(opt_res["dP"],opt_res["Tav"])

ax.set(xlabel='Pressure(Pa)', ylabel='Temperature (°C)',
        title='Pareto front and optimal solution')
ax.grid()
ax.legend(["Pareto front","Optimal solution"])


fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("Optimization results")

dk_res_T = res.goal_on_parameter("dk","Tav")
dk_res_P = res.goal_on_parameter("dk","dP")

nk_res_T = res.goal_on_parameter("nk","Tav")
nk_res_P = res.goal_on_parameter("nk","dP")

ax1.set(xlabel='dk (m)', ylabel='Temperature (°C)',
        title='Solutions for duct diameter dk')
ax1.grid()

ax1.scatter(dk_res_T[0],dk_res_T[1], c = "k")
ax11 = ax1.twinx()
ax11.set(ylabel='Pressure (Pa)')
ax11.scatter(dk_res_P[0],dk_res_P[1])

ax11.legend(["Pressure (Pa)"], loc = "upper left")
ax1.legend(["Temperature (°C)"], loc = "upper right")

ax2.scatter(nk_res_T[0],nk_res_T[1], c = "k")
ax2.set(xlabel='nk (-)', ylabel='Temperature (°C)',
        title='Solutions for duct number nk')
ax2.grid()
ax22 = ax2.twinx()
ax22.set( ylabel='Pressure (Pa)')
ax22.scatter(nk_res_P[0],nk_res_P[1])
ax22.legend(["Pressure (Pa)"], loc = "upper left")
ax2.legend(["Temperature (°C)"], loc = "upper right")
