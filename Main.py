import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skfuzzy import control as ctrl
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()
x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])
target = iris.target
iris = iris.data[:, :]


# print('x.shape: ', x.shape)
# print('y.shape: ', y.shape)
#
# print('x.head(5)', x.head())
# print('y.head(5)', y.head())
#
# print('x.describe()', x.describe())
# print('y.describe()', y.describe())


def init(x, display):
    import skfuzzy as fuzz

    # New Antecedent/Consequent objects hold universe variables and membership
    # functions1
    sepal_length = ctrl.Antecedent(np.arange(4, 8.1, 0.1), 'sepal_length')
    sepal_width = ctrl.Antecedent(np.arange(2, 5.1, 0.1), 'sepal_width')
    petal_length = ctrl.Antecedent(np.arange(1, 7.1, 0.1), 'petal_length')
    petal_width = ctrl.Antecedent(np.arange(0, 3.1, 0.1), 'petal_width')

    species = ctrl.Consequent(np.arange(0, 4, 1), 'species')

    # Auto-membership function population is possible with .automf(3, 5, or 7)
    sepal_length['small'] = fuzz.trimf(sepal_length.universe, [4, 4, x[0]])
    sepal_length['mid'] = fuzz.trimf(sepal_length.universe, [4, x[1], 8])
    sepal_length['big'] = fuzz.trimf(sepal_length.universe, [x[2], 8, 8])

    sepal_width['small'] = fuzz.trimf(sepal_width.universe, [2, 2, x[3]])
    sepal_width['mid'] = fuzz.trimf(sepal_width.universe, [2, x[4], 5])
    sepal_width['big'] = fuzz.trimf(sepal_width.universe, [x[5], 5, 5])

    petal_length['small'] = fuzz.trimf(petal_length.universe, [1, 1, x[6]])
    petal_length['mid'] = fuzz.trimf(petal_length.universe, [1, x[7], 7])
    petal_length['big'] = fuzz.trimf(petal_length.universe, [x[8], 7, 7])

    petal_width['small'] = fuzz.trimf(petal_width.universe, [0, 0, x[9]])
    petal_width['mid'] = fuzz.trimf(petal_width.universe, [0, x[10], 3])
    petal_width['big'] = fuzz.trimf(petal_width.universe, [x[11], 3, 3])

    # Custom membership functions can be built interactively with a familiar,
    # Pythonic API
    species['setosa'] = fuzz.trimf(species.universe, [0, 0, x[12]])
    species['versicolour'] = fuzz.trimf(species.universe, [0, x[13], 3])
    species['virginica'] = fuzz.trimf(species.universe, [x[14], 3, 3])

    # You can see how these look with .view()
    if display:
        sepal_length.view()
        sepal_width.view()
        petal_length.view()
        petal_width.view()
        plt.show()

    # 4 8
    # 2 5
    # 1 7
    # 0 3

    rules = []

    rules.append(ctrl.Rule(sepal_length['mid'], species['setosa']))
    rules.append(ctrl.Rule(sepal_width['mid'], species['setosa']))
    rules.append(ctrl.Rule(petal_length['small'], species['setosa']))
    rules.append(ctrl.Rule(petal_width['small'], species['setosa']))

    rules.append(ctrl.Rule(sepal_length['big'], species['versicolour']))
    rules.append(ctrl.Rule(sepal_width['small'], species['versicolour']))
    rules.append(ctrl.Rule(petal_length['mid'], species['versicolour']))
    rules.append(ctrl.Rule(petal_width['mid'], species['versicolour']))

    rules.append(ctrl.Rule(sepal_length['big'], species['virginica']))
    rules.append(ctrl.Rule(sepal_width['small'], species['virginica']))
    rules.append(ctrl.Rule(petal_length['big'], species['virginica']))
    rules.append(ctrl.Rule(petal_width['big'], species['virginica']))


    if display:
        rules[0].view()
        plt.show()





    tipping_ctrl = ctrl.ControlSystem(rules)
    tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

    # Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
    # Note: if you like passing many inputs all at once, use .inputs(dict_of_data)

    i = 0
    true = 0
    false = 0
    sef = 0
    set = 0
    vef = 0
    vet = 0
    vif = 0
    vit = 0
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
        if out < 1:
            # print('setosa')  # 0
            # species.view(sim=tipping)
            # plt.show()
            if target[i] == 0:
                true += 1
                set += 1
            else:
                false += 1
                sef += 1
        elif out < x[15]:
            # print('versicolour')  # 1
            # species.view(sim=tipping)
            # plt.show()
            if target[i] == 1:
                true += 1
                vet += 1
            else:
                false += 1
                vef += 1
        elif out < 3:
            # print('virginica')  # 2
            if target[i] == 2:
                true += 1
                vit += 1
            else:
                false += 1
                vif += 1
        i += 1
    # print("setosa t/f positiv: {}/{}, versicolour t/f positive: {}/{}, virginica t/f positive: {}/{}".format(set, sef, vet, vef, vit, vif))
    if display:
        species.view(sim=tipping)
        plt.show()

    return true, false

def fuzz(x, display):
    true, false = init(x, display)
    return false

#for drawing
results_ = list()
conv_list = list()
result_list = list()
def current_solution(curr_, convergence):
    # results_.append(curr_)
    # conv_list.append(convergence)
    result_list.append(fuzz(curr_, False))
    # print(convergence)
#---


from scipy.optimize import differential_evolution

bounds = [(4, 8)] * 3 + [(2, 5)] * 3 + [(1, 6)] * 3 + [(0, 3)] * 3 + [(0, 3)] + [(0, 3)] + [(0, 3)] + [(1, 2)]

result = differential_evolution(fuzz, bounds,
                                args=[False],  # don't display
                                maxiter=100, popsize=2,
                                tol=0.00000001,  # relative tolerance for convergence,
                                mutation=(0.1, 0.2), recombination=0.3,
                                workers=-1,  # parallel computing
                                disp=True,  # display status messages
                                updating='deferred',
                                # polish=False,  # L-BFGS-B method is used to polish the best population member at the
                                # end, which can improve the minimization slightly
                                callback=current_solution,  # callback for drawing)
                                )

print("Found solution at {} with value {}".format(result.x, result.fun))
# init(result.x, True) # display fuzzy rules and linguistic variables ba
plt.figure(figsize=(11, 4))
plots = plt.plot(result_list, 'c-')
plt.legend(plots, ('Wrong Answers',), frameon=True)
plt.xlabel('Iterations')
plt.grid()
plt.show()
