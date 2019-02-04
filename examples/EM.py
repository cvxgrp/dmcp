from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import cvxpy as cvx
import dmcp


def get_data(train_percent=1):
    '''
        Function that returns the dataset.
        input:
            train_percent:      The percentage of the iris dataset to be used
        output:
            partitioned_set:    The partitioned dataset in dictionary format. Dictionary has keys 'train' and 'test'.
                                Values are of the form (data, labels)
            data_dim:           The dimension of each data point
            num_classes:        The number of classes
            num_examples:       The number of data points for training
            
    '''
    # Get the iris dataset
    iris = datasets.load_iris()

    # Train Test Split
    train_set, test_set, train_labels, test_labels = train_test_split(iris.data, iris.target, test_size=1-train_percent, random_state=42)
    partitioned_set = {'train': (train_set, train_labels), 'test': (test_set, test_labels)}
    num_classes = len(set(train_labels))
    num_examples = len(train_labels)
    data_dim = len(train_set[0])

    return partitioned_set, data_dim, num_classes, num_examples


def get_variables(data_dim, num_classes, num_examples):
    '''
        Function that returns all the variables to the optimization problem.
        input:
            data_dim:               The dimension of each data point
            num_classes:            The number of classes
            num_examples:           The number of data examples
        output:
            variable_dict:          Dictionary of all the possible types of optimization variables.
                                    Key-value pairs can be any of the following:
                                        1) 'mean':          mean_vector
                                        2) 'precision':     precision_vector
                                        3) 'categorical':   categorical_vector
                                        4) 'conditional':   probability_matrix
                                    Values are described by the following:
                                        1) mean_vector:                 The object vector of all the means of each gaussian distribution in the mixture model
                                        2) precision_vector:            The object vector of all the precision matrices of each gaussian distribution in the mixture model
                                        3) categorical_vector:          The object vector of all the probabilities for the categorical distribution for the mixture model.
                                        4) probability_matrix:          The object matrix of all the conditional probabilities of the categorical distribution given the data 
                                                                        and the parameter estimates.
    '''
    # Vector of mean variables for each class
    mean_vector = np.asmatrix(np.zeros((num_classes,1), dtype=object))
    mean_vector2 = np.asmatrix(np.zeros((num_classes,1), dtype=object))
    for i in range(num_classes):
        mean_vector[i] = cvx.Variable((data_dim))
        mean_vector2[i] = cvx.Variable((data_dim))

    # Vector of inverse covariance matrices for each class
    precision_vector = np.asmatrix(np.zeros((num_classes,1), dtype=object))
    for i in range(num_classes):
        precision_vector[i] = cvx.Variable((data_dim,data_dim), PSD=True)

    # Vector of categorical distribution parameters for each class
    categorical_vector = np.asmatrix(np.zeros((num_classes,1), dtype=object))
    for i in range(num_classes):
        categorical_vector[i] = cvx.Variable(1, nonneg=True)

    # Define categorical distribution conditional probabilities
    probability_matrix = np.asmatrix(np.zeros((num_examples, num_classes), dtype=object))
    inverse_probability_matrix = np.asmatrix(np.zeros((num_examples, num_classes), dtype=object))
    for i in range(num_examples):
        for j in range(num_classes):
            probability_matrix[i,j] = cvx.Variable(1, nonneg=True)
            inverse_probability_matrix[i,j] = cvx.Variable(1, nonneg=True)
    
    variable_dict = {'mean': mean_vector, 'mean2': mean_vector2, 'precision': precision_vector, 'categorical': categorical_vector, 'conditional': probability_matrix, 'inverse_conditional': inverse_probability_matrix}
    return variable_dict


def get_objective(partitioned_set, variable_dict, data_dim, num_classes, num_examples):
    '''
        Function that returns the objective function object for the DMCP implementation of the EM algorithm.
        Idea is to split the objective function into a sum by the law of logarithms.
        input:
            partitioned_set:        Partitioned dataset.
            variable_dict:          Dictionary of all the possible types of optimization variables.
            data_dim:               The dimension of each data point
            num_classes:            The number of classes
            num_examples:           The number of data examples             
        output:
            objective:              Objective function
    '''
    # Get the dataset
    train_set = partitioned_set['train'][0]

    # Terms that only rely on the number of classes
    precision_log_det_matrix = np.asmatrix(np.zeros((num_classes,1), dtype=object)) # First term in the objective
    log_categorical_matrix = np.asmatrix(np.zeros((num_classes,1), dtype=object)) # Log categorical variables
    for i in range(num_classes):
        precision_log_det_matrix[i] = ((2*np.pi)**(-data_dim/2))*cvx.power(cvx.log_det(variable_dict['precision'][i,0]), 1/2)
        log_categorical_matrix[i] = cvx.log(variable_dict['categorical'][i,0])

    # Terms that change with both number of classes and number of examples
    log_normal_matrix = np.asmatrix(np.zeros((num_examples, num_classes), dtype=object)) # The log normal distribution matrix
    log_probability_matrix = np.asmatrix(np.zeros((num_examples, num_classes), dtype=object)) # The matrix of log categorical distribution conditional probabilities
    for i in range(num_examples):
        for j in range(num_classes):
            log_normal_matrix[i,j] = (-1/2)*((np.array(train_set[i]) - variable_dict['mean'][j,0]).T)*variable_dict['precision'][j,0]*(np.array(train_set[i]) - variable_dict['mean2'][j,0])
            log_probability_matrix[i,j] = cvx.log(variable_dict['inverse_conditional'][i,j])
    
    # Collect each term of the objective sum
    objective_array = []
    for i in range(num_examples):
        for j in range(num_classes):
            dummy = (variable_dict['conditional'][i,j])*(precision_log_det_matrix[j,0] + log_normal_matrix[i,j] + log_categorical_matrix[j,0] + log_probability_matrix[i,j])
            objective_array.append(dummy)

    # Define objective
    objective = cvx.Maximize(sum(objective_array))
    return objective


def get_constraints(partitioned_set, variable_dict, data_dim, num_classes, num_examples):
    '''
        Function that returns the list of constraints for the optimization problem.
        input:
            data_dim:               The dimension of each data point
            num_classes:            The number of classes
            num_examples:           The number of data examples 
        output:
            constraints:        List of constraints
    '''
    # Define constraints
    constraints = []

    # Categorical constraints
    categorical_array = []
    for i in range(num_classes):
        constraints.append(variable_dict['categorical'][i,0] >= 0)
        categorical_array.append(variable_dict['categorical'][i, 0])
    constraints.append(sum(categorical_array) == 1)

    # Conditional probability constraints
    for i in range(num_examples):
        conditional_array = []
        for j in range(num_classes):
            constraints.append(variable_dict['conditional'][i,j] >= 0)
            conditional_array.append(variable_dict['conditional'][i,j])
        constraints.append(sum(conditional_array) == 1)
    
    # Set mean and mean 2 to be equal
    for i in range(num_classes):
        constraints.append(variable_dict['mean'][i,0] == variable_dict['mean2'][i,0])
    
    # Set conditional matrices to be equal
    for i in range(num_examples):
        for j in range(num_classes):
            constraints.append(variable_dict['conditional'][i,j]*variable_dict['inverse_conditional'][i,j] == 1)

    return constraints


partitioned_set, data_dim, num_classes, num_examples = get_data(train_percent=0.01)
variable_dict = get_variables(data_dim, num_classes, num_examples)
objective = get_objective(partitioned_set, variable_dict, data_dim, num_classes, num_examples)
constraints = get_constraints(partitioned_set, variable_dict, data_dim, num_classes, num_examples)
problem = cvx.Problem(objective, constraints)
fix_var = [var[0,0] for var in variable_dict.values() if not var == variable_dict['inverse_conditional']]
fix_var2 = [var[0,0] for var in variable_dict.values() if not var == variable_dict['precision']]
print("Variable ID of Conditional is", variable_dict['inverse_conditional'][0,0].id)
print("Variable ID of Precision is", variable_dict['precision'][0,0].id)
print("Fix_var is", [var.id for var in fix_var])
print("Conditional Is_DMCP", dmcp.is_dmcp(dmcp.fix(problem, fix_var)))
print("Conditional Is_DCP", dmcp.fix(problem, fix_var).is_dcp())
print("Precision Is_DMCP", dmcp.is_dmcp(dmcp.fix(problem, fix_var2)))
print("Precision Is_DCP", dmcp.fix(problem, fix_var2).is_dcp())
print("Objective DMCP", dmcp.is_dmcp(objective.expr))
print("Is Problem DMCP? ", dmcp.is_dmcp(problem))
print("Number of variables: ", len(problem.variables()))