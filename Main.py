import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn import datasets
from matplotlib import pyplot as plt

# import some data to play with
iris = datasets.load_iris()
x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])
target = iris.target
iris = iris.data[:, :]

print('x.shape: ', x.shape)
print('y.shape: ', y.shape)

print('x.head(5)', x.head())
print('y.head(5)', y.head())

print('x.describe()', x.describe())


def init(x):
    import skfuzzy as fuzz

    # New Antecedent/Consequent objects hold universe variables and membership
    # functions1
    sepal_length = ctrl.Antecedent(np.arange(4, 8.1, 0.1), 'sepal_length')
    sepal_width = ctrl.Antecedent(np.arange(2, 5, 0.1), 'sepal_width')
    petal_length = ctrl.Antecedent(np.arange(1, 7, 0.1), 'petal_length')
    petal_width = ctrl.Antecedent(np.arange(0, 3, 0.1), 'petal_width')

    species = ctrl.Consequent(np.arange(0, 3, 1), 'species')


    # Auto-membership function population is possible with .automf(3, 5, or 7)
    sepal_length['small'] = fuzz.trimf(sepal_length.universe, [4, 4, x[0]])
    sepal_length['mid'] = fuzz.trimf(sepal_length.universe, [4, x[1], 8])
    sepal_length['big'] = fuzz.trimf(sepal_length.universe, [x[2], 8, 8])
    sepal_width.automf(3)
    petal_length.automf(3)
    petal_width.automf(3)

    # Differential
    # Evolution(DE)
    # do
    # ustalenia
    # parametrów
    # funkcji
    # trójkątnych
    # zamiast automatu powyżej

    # sepal_length['small'] = fuzz.trimf(sepal_length, [0 , 0, ])

    # Custom membership functions can be built interactively with a familiar,
    # Pythonic API
    species['setosa'] = fuzz.trimf(species.universe, [0, 0, 0])
    species['versicolour'] = fuzz.trimf(species.universe, [1, 1, 1])
    species['virginica'] = fuzz.trimf(species.universe, [2, 2, 2])

    # You can see how these look with .view()
    # sepal_length.view()
    # sepal_width.view()
    # petal_length.view()
    # petal_width.view()

    rule1 = ctrl.Rule(sepal_length['small'], species['setosa'])
    # rule1.view()
    # plt.show()

    rule2 = ctrl.Rule(sepal_length['mid'] | sepal_length['big'], species['versicolour'])
    rule3 = ctrl.Rule(sepal_length['small'], species['virginica'])

    rule4 = ctrl.Rule(sepal_width['good'], species['setosa'])
    rule5 = ctrl.Rule(petal_length['poor'], species['versicolour'])
    rule6 = ctrl.Rule(petal_width['average'], species['virginica'])

    rules=[]
    rules.append(rule1)
    rules.append(rule2)
    rules.append(rule3)
    rules.append(rule4)
    rules.append(rule5)
    rules.append(rule6)

    tipping_ctrl = ctrl.ControlSystem(rules)
    tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

    # Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
    # Note: if you like passing many inputs all at once, use .inputs(dict_of_data)



    i = 0
    true = 0
    false = 0
    for example in iris:
        # print('sepal_length: {}, sepal_width: {}, petal_length: {}, petal_width: {}'.format(
        #     example[0], example[1], example[2], example[3]))
        tipping.input['sepal_length'] = example[0]
        tipping.input['sepal_width'] = example[1]
        tipping.input['petal_length'] = example[2]
        tipping.input['petal_width'] = example[3]

        # Crunch the numbers
        tipping.compute()
        out = tipping.output['species']
        # print(out)

        if out < 1:
            # print('setosa')  # 0
            if target[i] == 0:
                true += 1
            else:
                false += 1
        elif out < 2:
            # print('versicolour')  # 1
            if target[i] == 1:
                true += 1
            else:
                false += 1
        else:
            # print('virginica')  # 2
            if target[i] == 2:
                true += 1
            else:
                false += 1

    i += 1

    # species.view(sim=tipping)
    # plt.show()

    return true, false


# true, false = init(5)
#
# print()
# print('true: {}, false: {}'.format(true, false))

def fuzz(x):

    true, false = init(x)
    return false

#for drawing
results_ = list()
def current_solution(curr_, convergence):
    # results_.append(curr_)
    print(curr_, convergence)
#---


from scipy.optimize import rosen, differential_evolution
bounds = [(4,8)]*3
result = differential_evolution(fuzz, bounds,
                                maxiter=25, popsize=25, tol=0.01, mutation=(0.5, 1), recombination=0.7, workers=-1, callback=current_solution) #callback for drawing)

#
print("Found solution at {} with value {}".format(result.x, result.fun))