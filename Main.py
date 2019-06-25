import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from skfuzzy import control as ctrl
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()
features = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])
# details of data set
print('features shape: ', features.shape)
print('target shape: ', y.shape)

print('features head(5)\n', features.head())
print('target head(5)\n', y.head())

print('features describe\n', features.describe())
print('target describe\n', y.describe())

target = iris.target
iris = iris.data[:, :]

step_eval_list = list()
mean_pop_eval_list = list()
visited = list()


# objective function to be minimized. Must be in the form f(x, *args), where x is the argument in the form of a 1-D
# array and args is a tuple of any additional fixed parameters needed to completely specify the function. create
# fuzzy logic controller and evaluate it
def fuzz(x, display):
    # this import must be inside of function for differential_evolution to work
    import skfuzzy as fuzz

    # New Antecedent/Consequent objects hold universe variables and membership
    # functions
    sepal_length = ctrl.Antecedent(np.arange(4, 8.1, 0.1), 'sepal_length')
    sepal_width = ctrl.Antecedent(np.arange(2, 5.1, 0.1), 'sepal_width')
    petal_length = ctrl.Antecedent(np.arange(1, 7.1, 0.1), 'petal_length')
    petal_width = ctrl.Antecedent(np.arange(0, 3.1, 0.1), 'petal_width')

    species = ctrl.Consequent(np.arange(0, 4, 1), 'species')

    # adding penalty for wrong sequence of parameters
    penalty = 0
    small_penalty = 1000
    distance_s_l = 0.4
    distance_s_w = 0.3
    distance_p_l = 0.4
    distance_p_w = 0.3
    distance_s = 0.1
    distance = 1.8
    if x[1] - x[0] < distance_s_l: penalty += small_penalty
    if x[2] - x[1] < distance_s_l: penalty += small_penalty
    if x[2] - x[0] < distance: penalty += small_penalty

    if x[4] - x[3] < distance_s_l: penalty += small_penalty
    if x[5] - x[4] < distance_s_l: penalty += small_penalty
    if x[5] - x[3] < distance: penalty += small_penalty

    if x[7] - x[6] < distance_s_l: penalty += small_penalty
    if x[8] - x[7] < distance_s_l: penalty += small_penalty
    if x[8] - x[6] < distance: penalty += small_penalty

    if x[10] - x[9] < distance_s_w: penalty += small_penalty
    if x[11] - x[10] < distance_s_w: penalty += small_penalty
    if x[11] - x[9] < distance: penalty += small_penalty

    if x[13] - x[12] < distance_s_w: penalty += small_penalty
    if x[14] - x[13] < distance_s_w: penalty += small_penalty
    if x[14] - x[12] < distance: penalty += small_penalty

    if x[16] - x[15] < distance_s_w: penalty += small_penalty
    if x[17] - x[16] < distance_s_w: penalty += small_penalty
    if x[17] - x[15] < distance: penalty += small_penalty

    if x[19] - x[18] < distance_p_l: penalty += small_penalty
    if x[20] - x[19] < distance_p_l: penalty += small_penalty
    if x[20] - x[18] < distance: penalty += small_penalty

    if x[22] - x[21] < distance_p_l: penalty += small_penalty
    if x[23] - x[22] < distance_p_l: penalty += small_penalty
    if x[23] - x[21] < distance: penalty += small_penalty

    if x[25] - x[24] < distance_p_l: penalty += small_penalty
    if x[26] - x[25] < distance_p_l: penalty += small_penalty
    if x[26] - x[24] < distance: penalty += small_penalty

    if x[28] - x[27] < distance_p_w: penalty += small_penalty
    if x[29] - x[28] < distance_p_w: penalty += small_penalty
    if x[29] - x[27] < distance: penalty += small_penalty

    if x[31] - x[30] < distance_p_w: penalty += small_penalty
    if x[32] - x[31] < distance_p_w: penalty += small_penalty
    if x[32] - x[30] < distance: penalty += small_penalty

    if x[34] - x[33] < distance_p_w: penalty += small_penalty
    if x[35] - x[34] < distance_p_w: penalty += small_penalty
    if x[35] - x[33] < distance: penalty += small_penalty

    if x[37] - x[36] < distance_s: penalty += small_penalty
    if x[38] - x[37] < distance_s: penalty += small_penalty
    if x[38] - x[36] < distance: penalty += small_penalty

    if x[40] - x[39] < distance_s: penalty += small_penalty
    if x[41] - x[40] < distance_s: penalty += small_penalty
    if x[41] - x[39] < distance: penalty += small_penalty

    if x[43] - x[42] < distance_s: penalty += small_penalty
    if x[44] - x[43] < distance_s: penalty += small_penalty
    if x[44] - x[42] < distance: penalty += small_penalty

    if penalty > 0:
        # print("penalty: ", penalty)
        step_eval_list.append(penalty)
        return penalty

    # Auto-membership function population is possible with .automf(3, 5, or 7)
    # Custom membership functions can be built interactively with a familiar,
    # Pythonic API
    sepal_length['small'] = fuzz.trimf(sepal_length.universe, [x[0], x[1], x[2]])
    sepal_length['mid'] = fuzz.trimf(sepal_length.universe, [x[3], x[4], x[5]])
    sepal_length['big'] = fuzz.trimf(sepal_length.universe, [x[6], x[7], x[8]])

    sepal_width['small'] = fuzz.trimf(sepal_width.universe, [x[9], x[10], x[11]])
    sepal_width['mid'] = fuzz.trimf(sepal_width.universe, [x[12], x[13], x[14]])
    sepal_width['big'] = fuzz.trimf(sepal_width.universe, [x[15], x[16], x[17]])

    petal_length['small'] = fuzz.trimf(petal_length.universe, [x[18], x[19], x[20]])
    petal_length['mid'] = fuzz.trimf(petal_length.universe, [x[21], x[22], x[23]])
    petal_length['big'] = fuzz.trimf(petal_length.universe, [x[24], x[25], x[26]])

    petal_width['small'] = fuzz.trimf(petal_width.universe, [x[27], x[28], x[29]])
    petal_width['mid'] = fuzz.trimf(petal_width.universe, [x[30], x[31], x[32]])
    petal_width['big'] = fuzz.trimf(petal_width.universe, [x[33], x[34], x[35]])

    species['setosa'] = fuzz.trimf(species.universe, [x[36], x[37], x[38]])
    species['versicolour'] = fuzz.trimf(species.universe, [x[39], x[40], x[41]])
    species['virginica'] = fuzz.trimf(species.universe, [x[42], x[43], x[44]])

    # display initial solution
    if not len(visited):
        display = 1
        visited.append(1)
    # You can see how these look with .view()
    if display:
        sepal_length.view()
        sepal_width.view()
        petal_length.view()
        petal_width.view()
        plt.show()

    # define the fuzzy relationship between input and output variables
    # simple rules based on intuition
    rules = [
        ctrl.Rule(sepal_length['mid'], species['setosa']),
        ctrl.Rule(sepal_width['mid'], species['setosa']),
        ctrl.Rule(petal_length['small'], species['setosa']),
        ctrl.Rule(petal_width['small'], species['setosa']),

        ctrl.Rule(sepal_length['big'], species['versicolour']),
        ctrl.Rule(sepal_width['small'], species['versicolour']),
        ctrl.Rule(petal_length['mid'], species['versicolour']),
        ctrl.Rule(petal_width['mid'], species['versicolour']),

        ctrl.Rule(sepal_length['big'], species['virginica']),
        ctrl.Rule(sepal_width['small'], species['virginica']),
        ctrl.Rule(petal_length['big'], species['virginica']),
        ctrl.Rule(petal_width['big'], species['virginica'])
    ]

    # You can see how rules look with .view()

    if display:
        rules[0].view()
        plt.show()

    # create control system
    classification_ctrl = ctrl.ControlSystem(rules)
    # in order to simulate we crate object representing our controller applied to a specific set of circumstances
    classification = ctrl.ControlSystemSimulation(classification_ctrl)

    i = 0
    true_positive = 0
    false_positive = 0
    setosa_false_positive = 0
    setosa_true_positive = 0
    versicolour_false_positive = 0
    versicolour_true_positive = 0
    virginica_false_positive = 0
    virginica_true_positive = 0

    # classify all samples from data set
    for sample in iris:
        # print('sepal_length: {}, sepal_width: {}, petal_length: {}, petal_width: {}'.format(
        #     sample[0], sample[1], sample[2], sample[3]))

        # Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
        # Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
        classification.input['sepal_length'] = sample[0]
        classification.input['sepal_width'] = sample[1]
        classification.input['petal_length'] = sample[2]
        classification.input['petal_width'] = sample[3]

        # Crunch the numbers
        classification.compute()

        out = classification.output['species']

        # assign a category
        if out < x[45]:
            if target[i] == 0:
                true_positive += 1
                setosa_true_positive += 1
            else:
                false_positive += 1
                setosa_false_positive += 1
        elif out < x[46]:
            if target[i] == 1:
                true_positive += 1
                versicolour_true_positive += 1
            else:
                false_positive += 1
                versicolour_false_positive += 1
        else:
            if target[i] == 2:
                true_positive += 1
                virginica_true_positive += 1
            else:
                false_positive += 1
                virginica_false_positive += 1
        i += 1

    # print("setosa t/f positiv: {}/{}, versicolour t/f positive: {}/{}, virginica t/f positive: {}/{}, "
    #       "false_positive: {}".format(setosa_true_positive, setosa_false_positive, versicolour_true_positive,
    #                                   versicolour_false_positive, virginica_true_positive, virginica_false_positive,
    #                                   false_positive))
    if display:
        species.view(sim=classification)
        plt.show()

    # as evaluation score to minimization return false_positive predictions
    step_eval_list.append(false_positive)
    return false_positive


# for drawing
results_ = list()
conv_list = list()
result_list = list()


def calculate_mean_pop_eval():
    eval_sum = 0
    for eval in step_eval_list:
        eval_sum += eval
    mean = eval_sum / len(step_eval_list)
    step_eval_list.clear()
    return mean

def current_solution(curr_, convergence):
    # results_.append(curr_)
    # conv_list.append(convergence)
    result_list.append(fuzz(curr_, False))
    mean = calculate_mean_pop_eval()
    print("mean: {0:.2f}".format(mean))
    mean_pop_eval_list.append(mean)
    # print(convergence)


# ---

# bounds for variables
# first line - membership functions antecedents
# second line - membership functions consequent
# third line - assign category
bounds = [(4, 8)] * 9 + [(2, 5)] * 9 + [(1, 6)] * 9 + [(0, 3)] * 9 \
         + [(0, 3)] * 9 \
         + [(0, 3)] + [(0, 3)]
print("\nDifferential evolution begins")
result = differential_evolution(fuzz, bounds,
                                args=[False],  # additional fixed parameters needed to completely specify the
                                # objective function - don't display
                                maxiter=500,  # maximum number of generations over which entire population is evolved
                                # maximum function evaluations (maxiter + 1) * popsize * len(x)
                                popsize=1,  # a multiplier for setting the total population size.
                                # population has popsize * len(x) individuals
                                tol=0.01,  # relative tolerance for convergence,
                                mutation=(0.1, 1),  # if specified as a float it should be in the range [0, 2], if
                                # specified as a tuple (min, max) dithering is employed; dithering randomly changes
                                # the mutation constant on a generation by generation basis.
                                recombination=0.3,  # crossover probability
                                # workers=-1,  # parallel computing - brakes calculating mean
                                disp=True,  # display status messages
                                updating='deferred',  # with 'deferred',
                                # the best solution vector is updated once per generation
                                # polish=False,  # L-BFGS-B method is used to polish the best population member at the
                                # end, which can improve the minimization slightly
                                callback=current_solution,  # callback for drawing
                                )

print("Found solution at {} with value {}".format(result.x, result.fun))
fuzz(result.x, True)  # display fuzzy rules and linguistic variables
plt.figure(figsize=(11, 4))  # display plot
plots = plt.plot(result_list, 'c-', mean_pop_eval_list, 'b-')
plt.legend(plots, ('Max fitness', 'Mean fitness'), frameon=True)
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.grid()
plt.show()
