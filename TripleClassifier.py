########################################################################
# Jason Baumbach
#   
#       Combined Classifier:
#           K-Nearest Neighbor (KNN)
#           Plugin Linear Discriminant Function (LDF)
#           Niave Bayes (NB) - with weight function
#
# Note: this code is available on GitHub 
#   https://github.com/jatlast/TripleClassifier.git
#
########################################################################

# required for reading csv files to get just the header
import csv
# required for sqrt function in Euclidean Distance calculation
import math
# required for parsing data files
import re

# allow command line options
import argparse
parser = argparse.ArgumentParser(description="Perform k-Nearest Neighbor, plugin-LDF, and their combination to classify train and test sets of varying n-dimensional data.")
parser.add_argument("-k", "--kneighbors", type=int, choices=range(1, 30), default=3, help="number of nearest neighbors to use")
parser.add_argument("-ft", "--filetrain", default="../uci_hd_preprocessing/data/cleveland_smoke_uci+_normal_train.csv", help="training file name (and path if not in . dir)")
parser.add_argument("-fs", "--filetest", default="../uci_hd_preprocessing/data/cleveland_smoke_uci+_normal_test.csv", help="testing file name (and path if not in . dir)")
parser.add_argument("-tn", "--targetname", default="target", help="the name of the target attribute")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2, 3], default=0, help="increase output verbosity")
args = parser.parse_args()

#   -- KNN specific --
# compute Euclidean distance between any two given vectors with any length.
def EuclideanDistanceBetweenTwoVectors(vOne, vTwo):
    distance = 0
    v_one_len = len(vOne)
    v_two_len = len(vTwo)
    # vOne & vTwo must be of equal length
    if(v_one_len != v_two_len):
        print(f"Warning UD: {v_one_len} != {v_two_len}")
        return -1

    for p in range(0, v_one_len):
        distance += math.pow((abs(float(vOne[p]) - float(vTwo[p]))), 2)
    return math.sqrt(distance)

#   -- LDF specific --
# get the inner product (dot product) of two equal length vectors
def GetInnerProductOfTwoVectors(vOne, vTwo):
    product = 0
    v_one_len = len(vOne)
    v_two_len = len(vTwo)
    # vOne & vTwo must be of equal length
    if(v_one_len != v_two_len):
        print(f"Warning DP: {v_one_len} != {v_two_len}")
        return -1
    else:
        for i in range(0, v_one_len):
            product += float(vOne[i]) * float(vTwo[i])
    return product

# variables that are useful to pass around
variables_dict = {
    'training_file' : args.filetrain
    , 'testing_file' : args.filetest
    , 'verbosity' : args.verbosity
    , 'kneighbors' : args.kneighbors
    , 'sqrt_2_pi' : 2.50662827463
    # UCI Heart Disease specific - attribut universally ignored
    #   Note: if they exist they are represented by the pre-processing added "smoke" attribute
    , 'ignore_columns' : ['cigs', 'years']
    # variables to enable dynamic use/ignore of target attribute
    , 'target_col_name' : args.targetname
    , 'target_col_index_train' : 0
    , 'target_col_index_test' : 0
    , 'training_col_count' : 0
    , 'testing_col_count' : 0
    # algorithm variables summed and compared after testing (com = combined)
    , 'test_runs_count' : 0
    , 'com_knn_ldf_right' : 0
    , 'com_knn_right_ldf_wrong' : 0
    , 'com_ldf_right_knn_wrong' : 0
    , 'com_ldf_wrong_knn_right' : 0
    , 'com_knn_wrong_ldf_right' : 0
    , 'com_knn_ldf_wrong' : 0
    # total times LDF confidence = 0
    , 'ldf_confidence_zero' : 0
    # used to compute the min-max of LDF confidence
    , 'ldf_diff_min' : 1000
    , 'ldf_diff_max' : 0
    # Niave Bayes continuous threshold
    , 'nb_continuous_threshold' : 0.2
    # initialization of the three confusion matrices
    , 'classifiers' : ['knn', 'ldf', 'nb']
#    , 'classifiers' : ['ldf', 'nb', 'knn']
    , 'knn_confusion_matrix' : {
        'TN'    : 0
        , 'FP'  : 0
        , 'FN'  : 0
        , 'TP'  : 0
    }
    , 'ldf_confusion_matrix' : {
        'TN'    : 0
        , 'FP'  : 0
        , 'FN'  : 0
        , 'TP'  : 0
    }
    , 'nb_confusion_matrix' : {
        'TN'    : 0
        , 'FP'  : 0
        , 'FN'  : 0
        , 'TP'  : 0
    }
    , 'com_confusion_matrix' : {
        'TN'    : 0
        , 'FP'  : 0
        , 'FN'  : 0
        , 'TP'  : 0
    }
}

# Specific to UCI's Heart Disease data set which has two target columns: num (0-4) & target (0 or 1)
# Allows the code to dynamically ignore the column NOT specified in the command line
if args.targetname == 'num':
    variables_dict['ignore_columns'].append('target')
elif args.targetname == 'target':
    variables_dict['ignore_columns'].append('num')

# Read the command line specified CSV data files
def ReadFileDataIntoDictOfLists(sFileName, dDictOfLists):
    # read the file
    with open(sFileName) as csv_file:
        line_number = 0
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            dDictOfLists[line_number] = row
            line_number += 1

# dynamically determine target index for inclusion/exclusion when creating vectors for comparison
def AddTargetIndexesToVarDict(dDictOfLists, dVariables):
    # column cound can vary between train and test sets
    col_count = len(dDictOfLists[0])
    # save the column count to the specified train or test variable in the dictionary
    if dDictOfLists['type'] == 'training':
        dVariables['training_col_count'] = col_count
    elif dDictOfLists['type'] == 'testing':
        dVariables['testing_col_count'] = col_count
    else:
        # this should never happen
        print(f"Warning: type {dDictOfLists['type']} not recognized.")
    # loop through the header to find and save the target column index to the variables dictionary
    for col in range(0, col_count):
        # check if the column name matches the target name command line option
        if dDictOfLists[0][col] == dVariables['target_col_name']:
            # save the column index of the found target column name
            if dDictOfLists['type'] == 'training':
                dVariables['target_col_index_train'] = col
            elif dDictOfLists['type'] == 'testing':
                dVariables['target_col_index_test'] = col

# dynamically determine target types in the training file
def AddTargetTypesToVarDict(dTrainingData, dVariables):
    dVariables['target_types'] = {} # key = type & value = count
    # loop through the training set (ignoring the header)
    for i in range(1, len(dTrainingData) - 1):
        # check if target type has already been discovered and added to the target_types variable
        if dTrainingData[i][dVariables['target_col_index_train']] not in dVariables['target_types']:
            # set to 1 upon first discovery
            dVariables['target_types'][dTrainingData[i][dVariables['target_col_index_train']]] = 1
        else:
            # otherwise sum like instances
            dVariables['target_types'][dTrainingData[i][dVariables['target_col_index_train']]] += 1

# dynamically determine the attributes shared between the train and test sets
def AddSharedAttributesToVarDict(dTrainingData, dTestingData, dVariables):
    dVariables['shared_attributes'] = [] # list of header names identical across train and test headers
    # check which data set is larger then loop through the smaller on the outside
    if dVariables['training_col_count'] < dVariables['testing_col_count']:
        # the training set is smaller so loop through its header on the outside
        for i in dTrainingData[0]:
            # ignore the irrelevant columns hard-coded at the beginning of the program ("num" or "target" are added dynmacially) 
            if i not in dVariables['ignore_columns']:
                # loop through the testing set header
                for j in dTestingData[0]:
                    # append matching header name to the list of shared attributes
                    if i == j:
                        dVariables['shared_attributes'].append(i)
    # the testing set is smaller...
    else:
        # ...so loop through its header on the outside
        for i in dTestingData[0]:
            # ignore the irrelevant columns...
            if i not in dVariables['ignore_columns']:
                # loop through the training set header
                for j in dTrainingData[0]:
                    # append matching header name to the list of shared attributes
                    if i == j:
                        dVariables['shared_attributes'].append(i)
        
# create and return a vector of only the shared attributes (excludes target attributes)
def GetVectorOfOnlySharedAttributes(dDictOfLists, sIndex, dVariables):
    header_vec = [] # for debugging only
    return_vec = [] # vector containing only shared attributes of row values at sIndex 
    col_count = 0 # train or test column number
    # set the appropriate col_count by data set type
    if dDictOfLists['type'] == 'training':
        col_count = dVariables['training_col_count']
    elif dDictOfLists['type'] == 'testing':
        col_count = dVariables['testing_col_count']
    else:
        # this should never happen
        print(f"Warning: type {dDictOfLists['type']} not recognized.")
    # loop through the header of the passed in data set
    for i in range(0, col_count):
        # ignore the row data at the index of the training set target attribute
        if i == dVariables['target_col_index_train'] and dDictOfLists['type'] == 'training':
            continue
        # ignore the row data at the index of the testing set target attribute
        elif i == dVariables['target_col_index_test'] and dDictOfLists['type'] == 'testing':
            continue
        # loop through the shared attributes list
        for col in dVariables['shared_attributes']:
        # check if the passed in header name matches a shared attribbute
            if dDictOfLists[0][i] == col:
                # append the shared attribute value at row[sIndex] col[i]
                return_vec.append(dDictOfLists[sIndex][i])
                # store for debugging incorrectly matched attributes
                header_vec.append(col)

    # debugging info
    if dVariables['verbosity'] > 2:
        print(f"shared {dDictOfLists['type']}:{header_vec}")
    
    return return_vec

#   -- KNN specific --
# initialize a dictionary of list objects equal to the number of neighbors to test
def InitNeighborsDict(dVariables):
    dVariables['neighbors_dict'] = {}
    for i in range(1, dVariables['kneighbors'] + 1):
        dVariables['neighbors_dict'][i] = {'index' : -1, 'distance' : 1000, 'type' : ''}

#   -- KNN specific --
# Populate the k-nearest neighbors by comparing all training data with test data point
def PopulateNearestNeighborsDicOfIndexes(dTrainingData, vTestData, dVariables):
    distances = []  # for debugging only (store then sort all distances for comparison to the chosen distances)
    # Loop through the training set (sans header) to find the least distance(s)
    for i in range(1, len(dTrainingData) - 1):
        # create the training vector of only the shared attributes to compare to the passed in testing vector (vTestData)
        train_vec = GetVectorOfOnlySharedAttributes(dTrainingData, i, dVariables)
        # get the Euclidean distance between the test & train vectors
        EuclideanDistance = EuclideanDistanceBetweenTwoVectors(vTestData, train_vec)
        distances.append(EuclideanDistance) # for debugging only
        # reset neighbor tracking variables
        neighbor_max_index = -1 # index of neighbor furthest away
        neighbor_max_value = -1 # value of neighbor furthest away
        # Loop through the neighbors dict so the maximum stored is always replaced first
        for j in range(1, len(dVariables['neighbors_dict']) + 1):
            if dVariables['neighbors_dict'][j]['distance'] > neighbor_max_value:
                neighbor_max_value = dVariables['neighbors_dict'][j]['distance']
                neighbor_max_index = j
        # save the newest least distance over the greatest existing neighbor distance
        # compare the current Euclidean distance against the value of neighbor furthest away
        if EuclideanDistance < neighbor_max_value:
            # since current distance is less, replace neighbor with max distance with current distance info
            dVariables['neighbors_dict'][neighbor_max_index]['index'] = i
            dVariables['neighbors_dict'][neighbor_max_index]['distance'] = EuclideanDistance
            dVariables['neighbors_dict'][neighbor_max_index]['type'] = dTrainingData[i][dVariables['target_col_index_train']]

    # debugging: print least k-distances from all k-distances calculated for comparison with chosen neighbors
    if dVariables['verbosity'] > 2:
        distances.sort()
        print("least distances:")
        for i in range(0, len(dVariables['neighbors_dict'])):
            print(f"min{i}:({distances[i]}) \t& neighbors:({dVariables['neighbors_dict'][i+1]['distance']})")

#   -- KNN specific --
# Calculate majority type and confidence from the majority of nearest neighbors
def AddKNNMajorityTypeToVarDict(dVariables):
    type_count_dict = {} # store key = type & value = sum of neighbors with this type
    # zero out KNN majority type tracking variables
    dVariables['knn_best_target'] = 'UNK'
    dVariables['knn_majority_count'] = 0
    dVariables['knn_confidence'] = 0
    # loop through the target types and zero out the type_count_dict
    for key in dVariables['target_types']:
        type_count_dict[key] = 0

    # loop through the nearest neighbors and total the different type hits
    for i in range(1, len(dVariables['neighbors_dict']) + 1):
        type_count_dict[dVariables['neighbors_dict'][i]['type']] += 1

    # loop through the target types to set the majority info for KNN confidence calculation
    for key in type_count_dict:
        # current is better than best
        if dVariables['knn_majority_count'] < type_count_dict[key]:
            # set best to current
            dVariables['knn_majority_count'] = type_count_dict[key]
            dVariables['knn_best_target'] = key

    # calculate confidence as (majority / k-nearest neighbors)
    dVariables['knn_confidence'] = dVariables['knn_majority_count'] / len(dVariables['neighbors_dict'])

    # debug info
    if dVariables['verbosity'] > 2:
        print(f"majority:{dVariables['knn_best_target']}{type_count_dict}")

#   -- LDF specific --
# calculate the target type means from the training data
def AddTargetTypeMeansToVarDict(dTrainingData, dVariables):
    col_sums_dic = {} # [target][col_name] = sums by target
    row_count_dic = {} # [target] = row count by target

    # zero out the col_sums and row_count dictionaries
    for key in dVariables['target_types']:
        col_sums_dic[key] = {} # col_sums_dic[target][col_name] = sums by target
        row_count_dic[key] = 0 # row_count_dic[target] = row count by target
        # dynamically create target mean vectors for each target type to the variables dictionary for LDF calculations
        dVariables[key] = {'ldf_mean' : []} # initialized to the empty list
        # loop thought the sared attributes list
        for col in dVariables['shared_attributes']:
            if col != dVariables['target_col_name']:
                # initialize the column sum to zero since this is a shared attribute column
                col_sums_dic[key][col] = 0

    # Loop through the training set to calculate the totals required to calculate the means
    for i in range(1, len(dTrainingData) - 1): # loop through the traing set rows
        for j in range(0, len(dTrainingData[0])): # loop through the traing set columns
            for col in dVariables['shared_attributes']: # loop through the shared columns
                # check if the column is shared
                if dTrainingData[0][j] == col:
                    # only sum the non-target columns
                    if col != dVariables['target_col_name']:
                        # sum the colum values
                        col_sums_dic[dTrainingData[i][dVariables['target_col_index_train']]][col] += float(dTrainingData[i][j])
                    # use the target column as a que to increment the row count
                    else:
                        # incrament the row count
                        row_count_dic[dTrainingData[i][dVariables['target_col_index_train']]] += 1

    # dynamically calculate the appropriate number of target means
    for key in dVariables['target_types']: # loop through the target types
        for col in col_sums_dic[key]: # loop through the columns that were summed by target
            # debug info
            if dVariables['verbosity'] > 2:
                print(f"col:{col}:\t{col_sums_dic[key][col]} / {row_count_dic[key]}")
            # append the colum mean to the target type mean vector
            if row_count_dic[key] > 0:
                dVariables[key]['ldf_mean'].append(col_sums_dic[key][col] / row_count_dic[key])
            else:
                # this should never happen
                print(f"Warning: LDF mean = 0 for target:{key}")
                print(f"col:{col}:\t{col_sums_dic[key][col]} / {row_count_dic[key]}")
                dVariables[key]['ldf_mean'].append(0)

#   -- LDF specific --
# calculate the inner (dot) products of the different target type means
def AddMeanSqCalcsToVarDic(dVariables):
    # loop through the target types
    for key in dVariables['target_types']:
        # calculate the inner (dot) products of the different target type means
        dVariables[key]['ldf_mean_square'] = GetInnerProductOfTwoVectors(dVariables[key]['ldf_mean'], dVariables[key]['ldf_mean'])

#   -- LDF specific --
# calculate the largets and second largets g(x) to determine the target and confidence of the LDF function
def AddCalcsOfPluginLDFToVarDic(vTestData, dVariables):
    # initialize calculation variables
    dVariables['ldf_best_g'] = -1 # use -1 so best begins < least possible g(d)
    dVariables['ldf_second_best_g'] = -1 # use -1 so second best begins < least possible g(d)
    dVariables['ldf_best_target'] = 'UNK'
    dVariables['ldf_second_best_target'] = 'UNK'
    dVariables['ldf_confidence'] = 0
    ldf_diff = 0
    # loop through the target types
    for key in dVariables['target_types']:
        # calculate the inner (dot) products of the target type means
        dVariables[key]['ldf_dot_mean'] = GetInnerProductOfTwoVectors(vTestData, dVariables[key]['ldf_mean'])
        # calculate g(x)
        dVariables[key]['ldf_g'] = (2 * dVariables[key]['ldf_dot_mean']) - dVariables[key]['ldf_mean_square']

        # store the largest and second largest g(x) for later comparison to determine confidence
        # current better than second best
        if dVariables['ldf_second_best_g'] < dVariables[key]['ldf_g']:
            # current better than best
            if dVariables['ldf_best_g'] < dVariables[key]['ldf_g']:
                # set second best to previous best
                dVariables['ldf_second_best_g'] = dVariables['ldf_best_g']
                dVariables['ldf_second_best_target'] = dVariables['ldf_best_target']
                # set best to current
                dVariables['ldf_best_g'] = dVariables[key]['ldf_g']
                dVariables['ldf_best_target'] = key
            else:
                # set second best to current best
                dVariables['ldf_second_best_g'] = dVariables[key]['ldf_g']
                dVariables['ldf_second_best_target'] = key
        # current better than best
        elif dVariables['ldf_best_g'] < dVariables[key]['ldf_g']:
            # set second best to previous best
            dVariables['ldf_second_best_g'] = dVariables['ldf_best_g']
            dVariables['ldf_second_best_target'] = dVariables['ldf_best_target']
            # set best to current
            dVariables['ldf_best_g'] = dVariables[key]['ldf_g']
            dVariables['ldf_best_target'] = key

        # debug info: print the formul used
        if dVariables['verbosity'] > 2:
            print(f"\t{key}_g(x): {round(dVariables[key]['ldf_g'], 2)} = (2 * {round(dVariables[key]['ldf_dot_mean'], 2)}) - {round(dVariables[key]['ldf_mean_square'], 2)}")

    # get the difference between best and second best
    ldf_diff = dVariables['ldf_best_g'] - dVariables['ldf_second_best_g']

    # reset the max
    if dVariables['ldf_diff_max'] < ldf_diff:
        dVariables['ldf_diff_max'] = ldf_diff
    
    # reset the min
    if dVariables['ldf_diff_min'] > ldf_diff:
        dVariables['ldf_diff_min'] = ldf_diff
    
    # use min-max to calculate confidence if min & max have been initialized
    if dVariables['ldf_diff_max'] != dVariables['ldf_diff_min']:
        dVariables['ldf_confidence'] = ((ldf_diff - dVariables['ldf_diff_min']) / (dVariables['ldf_diff_max'] - dVariables['ldf_diff_min']))
    else:
        dVariables['ldf_confidence'] = ldf_diff

    # debugging: sum all LDF confidenc <= 0
    if dVariables['ldf_confidence'] < 0:
        dVariables['ldf_confidence_zero'] += 1
        if dVariables['verbosity'] > 2:
            print(f"ldf diff:{dVariables['ldf_best_g']} - {dVariables['ldf_second_best_g']}")

#   -- NB specific --
# calculate the target type probabilities & means by column from the training data
def AddProbabilitiesToVarDict(dTrainingData, dVariables):
    # used for calculating column means by target (used if col is continuous)
    col_sums_dic = {} # [target][col_name] = sums by target
    row_count_dic = {} # [target] = row count by target
    # used for calculating unique column value counts by target
    row_val_count_dic = {} # [target][col_name][col_val] = row value counts by target & column
    dVariables['continuous_columns'] = [] # list of continuous columns
    dVariables['target_type_probabilities'] = {} # dictionary of probabilities by type ['target'] = probability

    # zero out the col_sums and row_count dictionaries
    for key in dVariables['target_types']:
        col_sums_dic[key] = {} # col_sums_dic[target][col_name] = sums by target
        row_count_dic[key] = 0 # row_count_dic[target] = row count by target
        row_val_count_dic[key] = {} # row_val_count_dic[target][col_name][col_val] = row value counts by target & column
        # calculate and add the target type probability
        dVariables['target_type_probabilities'][key] = (dVariables['target_types'][key] / (len(dTrainingData) - 2))
        # loop thought the sared attributes list
        for col in dVariables['shared_attributes']:
            if col != dVariables['target_col_name']:
                # initialize the column sum to zero since this is a shared attribute column
                col_sums_dic[key][col] = 0
                row_val_count_dic[key][col] = {} # row_val_count_dic[target][col_name][col_val] = row value counts by target & column
                # dictionary variable to hold [target][col_name]['mean'] = mean
                dVariables[key][col] = {} # initialized to the empty dictionary

    # Loop through the training set to calculate the totals required to calculate the probabilities & means
    for i in range(1, len(dTrainingData) - 1): # loop through the traing set rows
        for j in range(0, len(dTrainingData[0])): # loop through the traing set columns
            for col in dVariables['shared_attributes']: # loop through the shared columns
                # check if the column is shared
                if dTrainingData[0][j] == col:
                    # only sum the non-target columns
                    if col != dVariables['target_col_name']:
                        # store the value in training data cell [i][j]
                        cell_value_orig = dTrainingData[i][j]
                        cell_value_float = float(dTrainingData[i][j])
                        # sum the colum values by target (col_sums_dic[target][col_name] += cell_value_float)
                        col_sums_dic[dTrainingData[i][dVariables['target_col_index_train']]][col] += cell_value_float
                        
                        # check if column is continuous (assuming data is normalized and cells containing decimal values < nb_continuous_threshold are continuous)
                        if col not in dVariables['continuous_columns'] and cell_value_float > 0 and cell_value_float < dVariables['nb_continuous_threshold']:
                            dVariables['continuous_columns'].append(col)

                        # check if cell_value_orig already in dictionary
                        if cell_value_orig in row_val_count_dic[dTrainingData[i][dVariables['target_col_index_train']]][col]:
                            # yes - incrament
                            row_val_count_dic[dTrainingData[i][dVariables['target_col_index_train']]][col][cell_value_orig] += 1
                        else:
                            # yes - initialize
                            row_val_count_dic[dTrainingData[i][dVariables['target_col_index_train']]][col][cell_value_orig] = 1
                    # use the target column as a que to increment the row count
                    else:
                        # incrament the row count
                        row_count_dic[dTrainingData[i][dVariables['target_col_index_train']]] += 1

    # dynamically calculate the appropriate number of target probablities & means
    for key in dVariables['target_types']: # loop through the target types
        for col in dVariables['shared_attributes']: # loop through the shared columns
            if col != dVariables['target_col_name']:
                # add the colum mean to the variables dict by target type
                if row_count_dic[key] > 0:
                    dVariables[key][col]['mean'] = col_sums_dic[key][col] / row_count_dic[key]
                    # debug info
                    if dVariables['verbosity'] > 2:
                        print(f"target:{key} | col:{col} |\t{col_sums_dic[key][col]} / {row_count_dic[key]}")
                else:
                    # this should never happen
                    print(f"Warning: NB mean = 0 for target:{key}")
                    print(f"col:{col}:\t{col_sums_dic[key][col]} / {row_count_dic[key]}")
                    dVariables[key][col]['mean'] = 0
                # loop through the val caount dict to calculate target|value probabilities
                for val in row_val_count_dic[key][col]:
                    if val != 'mean': # ignore the key[mean] added above
                        # debug info
                        if dVariables['verbosity'] > 2:
                            print(f"target:{key} | col:{col} | val:{val}")
                        if val not in dVariables[key][col]:
                            # calculate and add probability of [target][col][val] to dictionary as [target][col][val][probability]
                            dVariables[key][col][val] = {'probability' : (row_val_count_dic[key][col][val] / dVariables['target_types'][key]) }
                        else:
                            print(f"Warning: This should never happen: val{val} already in dictionary")

#   -- NB specific --
# calculate standard deviations for continuous columns
def AddStandardDeviationsToVarDict(dTrainingData, dVariables):
    col_diff_mean_sq_sums_dic = {} # [target][col_name] = sums of sqrt(sq(val - mean))
    row_count_dic = {} # [target] = row count by target
    # zero out the dictionary
    for key in dVariables['target_types']:
        col_diff_mean_sq_sums_dic[key] = {} # col_diff_mean_sq_sums_dic[target][col_name] = sums of sqrt(sq(val - mean))
        row_count_dic[key] = 0 # row_count_dic[target] = row count by target
        # loop though the continuous columns list
        for col in dVariables['continuous_columns']:
            col_diff_mean_sq_sums_dic[key][col] = 0 # initialized to zero

    # Loop through the training set to calculate the totals required to calculate the probabilities & means
    for i in range(1, len(dTrainingData) - 1): # loop through the traing set rows
        first_col = True # used to increment the row_count
        for j in range(0, len(dTrainingData[0])): # loop through the traing set columns
            for col in dVariables['continuous_columns']: # loop through the continuous columns
                # check if the column is one of the continuous columns
                if dTrainingData[0][j] == col:
                    # append the result of subtracting the Mean from the value and squaring the result
                    col_diff_mean_sq_sums_dic[dTrainingData[i][dVariables['target_col_index_train']]][col] += math.pow( ( float(dTrainingData[i][j]) - dVariables[dTrainingData[i][dVariables['target_col_index_train']]][col]['mean'] ), 2)
                    # increment the rowcount upon the first iteration
                    if first_col == True:
                        first_col = False
                        row_count_dic[dTrainingData[i][dVariables['target_col_index_train']]] += 1

    # dynamically calculate the appropriate number of target standard deviations
    for key in dVariables['target_types']: # loop through the target types
        for col in dVariables['continuous_columns']:
            # add the colum mean to the variables dict by target type
            if row_count_dic[key] > 0:
                dVariables[key][col]['stdev'] = math.sqrt(col_diff_mean_sq_sums_dic[key][col] / row_count_dic[key])
                # debug info
                if dVariables['verbosity'] > 2:
                    print(f"stdev - target:{key} | col:{col} |\t{col_diff_mean_sq_sums_dic[key][col]} / {row_count_dic[key]}")
            else:
                # this should never happen
                print(f"Warning: NB stdev = 0 for target:{key}")
                print(f"col:{col}:\t{col_diff_mean_sq_sums_dic[key][col]} / {row_count_dic[key]}")
                dVariables[key][col]['stdev'] = 0

    # debug info
    if dVariables['verbosity'] > 2:
        # loop through continuous column name
        for col in dVariables['continuous_columns']:
            for key in dVariables['target_types']: # loop through the target types
                print(f"col:{col} | target:{key} | mean:{dVariables[key][col]['mean']} | stdev:{dVariables[key][col]['stdev']}")

#   -- NB specific --
# calculate Niave Base probabilities for every target type and determine best
def GetContinuousGausianCalcValue(xVal, sColumnName, sTarget, dVariables):
    return_value = 0
    left_val = 1 / ( dVariables['sqrt_2_pi'] * dVariables[sTarget][sColumnName]['stdev'] )
    right_exponent = -math.pow(float(xVal) - dVariables[sTarget][sColumnName]['mean'], 2) / ( 2 * math.pow( dVariables[sTarget][sColumnName]['stdev'], 2) )
    right_val = math.exp(right_exponent)
    return_value = left_val * right_val
    # if return_value > 0:
    #     print(f"col:{sColumnName} | x:{xVal} | return_value:{return_value} = left:{left_val} | exponent:{right_exponent} | right:{right_val} | target:{sTarget} | mean:{dVariables[sTarget][sColumnName]['mean']} | stdev:{dVariables[sTarget][sColumnName]['stdev']} | sqrt_2_pi:{dVariables['sqrt_2_pi']}")
    return return_value

#   -- NB specific --
# calculate Niave Base probabilities for every target type and determine best
def AddTestProbabilityToVarDic(vTestData, dVariables):
    dVariables['nb_confidence_dict'] = {}
    dVariables['nb_best_p'] = -1 # use -1 so best begins < least possible probability
    dVariables['nb_second_best_p'] = -1 # use -1 so second best begins < least possible probability
    dVariables['nb_best_target'] = 'UNK'
    dVariables['nb_second_best_target'] = 'UNK'
    dVariables['nb_confidence'] = 0

    # zero out the dictionary
    for key in dVariables['target_types']:
        # initialize to P(target)
        dVariables['nb_confidence_dict'][key] = dVariables['target_type_probabilities'][key] # variables_dict[target] = P(X|target) * P(target)
    
    for i in range(0, len(vTestData)):
#        print(f"col:{dVariables['shared_attributes'][i]}")
        for key in dVariables['target_types']:
            if dVariables['shared_attributes'][i] not in dVariables['continuous_columns']:
                # running multiplication total of P(target) * P(x|target)

                # handle missing values between training and testing data sets
                if vTestData[i] in dVariables[key][dVariables['shared_attributes'][i]]:
                    dVariables['nb_confidence_dict'][key] *= dVariables[key][dVariables['shared_attributes'][i]][vTestData[i]]['probability']
                else:
                    # multiply by zero so confidence will be zero for this unknown value
                    dVariables['nb_confidence_dict'][key] *= 0
                    print(f"Warning: This should not happen: (col:{dVariables['shared_attributes'][i]} | target:{key}) test val:{vTestData[i]} not in prob dict:{dVariables[key][dVariables['shared_attributes'][i]]}")
            else:
                dVariables['nb_confidence_dict'][key] *= GetContinuousGausianCalcValue(vTestData[i], dVariables['shared_attributes'][i], key, dVariables)

    # loop through Naive Bayes confidence dictionary of target types
    for key in dVariables['nb_confidence_dict']:
#        print(f"target:{key} | confidence:{dVariables['nb_confidence_dict'][key]}")

        # store the largest and second largest confidence values
        # current better than second best
        if dVariables['nb_second_best_p'] < dVariables['nb_confidence_dict'][key]:
            # current better than best
            if dVariables['nb_best_p'] < dVariables['nb_confidence_dict'][key]:
                # set second best to previous best
                dVariables['nb_second_best_p'] = dVariables['nb_best_p']
                dVariables['nb_second_best_target'] = dVariables['nb_best_target']

                # set best to current
                dVariables['nb_best_p'] = dVariables['nb_confidence_dict'][key]
                dVariables['nb_best_target'] = key
            else:
                # set second best to current best
                dVariables['nb_second_best_p'] = dVariables['nb_confidence_dict'][key]
                dVariables['nb_second_best_target'] = key
        # current better than best
        elif dVariables['nb_best_p'] < dVariables['nb_confidence_dict'][key]:
            # set second best to previous best
            dVariables['nb_second_best_p'] = dVariables['nb_best_p']
            dVariables['nb_second_best_target'] = dVariables['nb_best_target']
            # set best to current
            dVariables['nb_best_p'] = dVariables['nb_confidence_dict'][key]
            dVariables['nb_best_target'] = key

    dVariables['nb_confidence'] = dVariables['nb_best_p']
#    print(f"NB: best target:{dVariables['nb_best_target']} | confidence:{dVariables['nb_best_p']} | 2nd best target:{dVariables['nb_second_best_target']} | confidence:{dVariables['nb_second_best_p']}")


# sum and track confusion matrix by sPrefix (i.e., knn, ild, com) and store in the variables dictionary
def TrackConfusionMatrixSums(sTestType, sPredictionType, sPrefix, dVariables):
    # target is positive
    if int(float(sTestType)) > 0: # unfortunately, assuming the target is numeric makes the code dependant on numerical target values
        # target matches prediction
        if sTestType == sPredictionType:
            # increment true positive (TP) count
            dVariables[sPrefix + '_confusion_matrix']['TP'] += 1
        # target does not match prediction
        else:
            # increment false positive (FP) count
            dVariables[sPrefix + '_confusion_matrix']['FP'] += 1
    # target is negative
    else:
        # target matches prediction
        if sTestType == sPredictionType:
            # increment true negative (TN) count
            dVariables[sPrefix + '_confusion_matrix']['TN'] += 1
        # target does not match prediction
        else:
            # increment false negative (FN) count
            dVariables[sPrefix + '_confusion_matrix']['FN'] += 1

# simply print the confusion matrix by sPrefix (i.e., knn, ild, com) along with calculated stats
def PrintConfusionMatrix(sPrefix, dVariables):
    # print the confusion matrix
    print(f"\n{sPrefix} - Confusion Matrix:\n\tTP:{dVariables[sPrefix + '_confusion_matrix']['TP']} | FN:{dVariables[sPrefix + '_confusion_matrix']['FN']}\n\tFP:{dVariables[sPrefix + '_confusion_matrix']['FP']} | TN:{dVariables[sPrefix + '_confusion_matrix']['TN']}")
    # Classifier Accuracy, or recognition rate: percentage of test set tuples that are correctly classified
    #   calculate accuracy = ( (TP + TN) / All )
    dVariables[sPrefix + '_accuracy'] = round((dVariables[sPrefix + '_confusion_matrix']['TP'] + dVariables[sPrefix + '_confusion_matrix']['TN']) / (dVariables[sPrefix + '_confusion_matrix']['TP'] + dVariables[sPrefix + '_confusion_matrix']['TN'] + dVariables[sPrefix + '_confusion_matrix']['FP'] + dVariables[sPrefix + '_confusion_matrix']['FN']),2)
    # calculate error rate = (1 - accuracy)
    dVariables[sPrefix + '_error_rate'] = round((1 - dVariables[sPrefix + '_accuracy']),2)
    # Sensitivity, or Recall, or True Positive recognition rate (TPR)
    #   Recall: completeness – what % of positive tuples the classifier label positive
    #   Note: 1.0 = Perfect score
    #       calculate sensitivity|recall|TPR = ( TP / (TP + FN) )
    dVariables[sPrefix + '_sensitivity'] = round(dVariables[sPrefix + '_confusion_matrix']['TP'] / (dVariables[sPrefix + '_confusion_matrix']['TP'] + dVariables[sPrefix + '_confusion_matrix']['FN']),2)
    # Precision: exactness – what % of tuples labeled positive are positive
    #   calculate precision = ( TP / (TP + FP) )
    dVariables[sPrefix + '_precision'] = round(dVariables[sPrefix + '_confusion_matrix']['TP'] / (dVariables[sPrefix + '_confusion_matrix']['TP'] + dVariables[sPrefix + '_confusion_matrix']['FP']),2)
    # Specificity, or True Negative recognition rate (TNR)
    #   calculate specificity|TNR = ( TN / (TN + FP) )
    dVariables[sPrefix + '_specificity'] = round(dVariables[sPrefix + '_confusion_matrix']['TN'] / (dVariables[sPrefix + '_confusion_matrix']['TN'] + dVariables[sPrefix + '_confusion_matrix']['FP']),2)
    # calculate false positive rate (FPR) = (1 - specificity) or ( FP / (FP + TN) )
    dVariables[sPrefix + '_FPR'] = round(dVariables[sPrefix + '_confusion_matrix']['FP'] / (dVariables[sPrefix + '_confusion_matrix']['FP'] + dVariables[sPrefix + '_confusion_matrix']['TN']),2)
    # F measure (F1 or F-score): harmonic mean of precision and recall
    #   calculate F-score = (2 * precision * recall) / (precision + recall)
    dVariables[sPrefix + '_fscore'] = round((2 * dVariables[sPrefix + '_precision'] * dVariables[sPrefix + '_sensitivity']) / (dVariables[sPrefix + '_precision'] + dVariables[sPrefix + '_sensitivity']),2)
    # print the values calculated above
    print(f"Accuracy   :{dVariables[sPrefix + '_accuracy']}")
    print(f"Error Rate :{dVariables[sPrefix + '_error_rate']}")
    print(f"Sensitivity:{dVariables[sPrefix + '_sensitivity']}")
    print(f"Precision  :{dVariables[sPrefix + '_precision']}")
    print(f"Specificity:{dVariables[sPrefix + '_specificity']}")
    print(f"FPR        :{dVariables[sPrefix + '_FPR']}")
    print(f"F-score    :{dVariables[sPrefix + '_fscore']}")

# keep track of the running totals of which algorithms were correct and/or incorrect
def AddRunningPredictionStatsToVarDict(sTestType, dVariables):
    dVariables['test_runs_count'] += 1
    # COM-bination right
    if sTestType == dVariables['com_best_target']:
        # KNN right
        if sTestType == dVariables['knn_best_target']:
            # LDF right
            if sTestType == dVariables['ldf_best_target']:
                dVariables['com_knn_ldf_right'] += 1
            # LDF wrong
            else:
                dVariables['com_knn_right_ldf_wrong'] += 1
        # KNN wrong
        else:
            # LDF right
            if sTestType == dVariables['ldf_best_target']:
                dVariables['com_ldf_right_knn_wrong'] += 1
            # LDF wrong
            else:
                # this should never happen!
                print("Warning: COM-bination can never be connrect when both KNN & LDF are incorrect.")
    # COM-bination wrong
    else:
        # KNN right
        if sTestType == dVariables['knn_best_target']:
            # LDF right
            if sTestType == dVariables['ldf_best_target']:
                # this should never happen!
                print("Warning: COM-bination can never be inconnrect when both KNN & LDF are correct.")
            # LDF wrong
            else:
                dVariables['com_ldf_wrong_knn_right'] += 1
        # KNN wrong
        else:
            # LDF right
            if sTestType == dVariables['ldf_best_target']:
                dVariables['com_knn_wrong_ldf_right'] += 1
            # LDF wrong
            else:
                dVariables['com_knn_ldf_wrong'] += 1

# Load the training data
training_dict = {'type' : 'training'} # set the type for dynamically determining shared attributes
# read the training csv into the training dict
ReadFileDataIntoDictOfLists(variables_dict['training_file'], training_dict)

# add the target indexes of the training set to the variables dictionary
AddTargetIndexesToVarDict(training_dict, variables_dict)

# add the possible target types to the variables dictionary
AddTargetTypesToVarDict(training_dict, variables_dict)
# print debugging info
if variables_dict['verbosity'] > 0:
    for key in variables_dict['target_types']:
        print(f"target types {key}:{variables_dict['target_types'][key]}")

# Load the testing data
testing_dict = {'type' : 'testing'} # set the type for dynamically determining shared attributes
# read the testing csv into the testing dict
ReadFileDataIntoDictOfLists(variables_dict['testing_file'], testing_dict)

# add the target indexes of the testing set to the variables dictionary
AddTargetIndexesToVarDict(testing_dict, variables_dict)

# add the shared attributes for comparing testing data with training data to the variables dictionary
AddSharedAttributesToVarDict(training_dict, testing_dict, variables_dict)
# debugging info
if variables_dict['verbosity'] > 0:
    # shared attribute includes "target"
    print(f"shared attributes:{variables_dict['shared_attributes']}")
    # vector attribute includes all shared attributes except the "target" attribute
    print(f"vector attributes:{GetVectorOfOnlySharedAttributes(testing_dict, 0, variables_dict)}")

#   -- LDF specific --
# add the target type means to the variables dictionary for later use
AddTargetTypeMeansToVarDict(training_dict, variables_dict)

#   -- LDF specific --
# calculate the inner (dot) products of the different target type means
AddMeanSqCalcsToVarDic(variables_dict)

# debugging info
if variables_dict['verbosity'] > 1:
    for key in variables_dict['target_types']:
        print(f"{key} mean_sq:{variables_dict[key]['ldf_mean_square']} | mean:{variables_dict[key]['ldf_mean']}")

#   -- KNN specific --
# initialize the dictionary of list objects equal to the number of neighbors to test
InitNeighborsDict(variables_dict)

# debug info
if variables_dict['verbosity'] > 0:
    print(f"neighbors: {variables_dict['kneighbors']} = {len(variables_dict['neighbors_dict'])} :len(neighbors_dict)")

#   -- NB specific --
AddProbabilitiesToVarDict(training_dict, variables_dict)

#   -- NB specific --
AddStandardDeviationsToVarDict(training_dict, variables_dict)

# debug info
if variables_dict['verbosity'] > 1:
    print("NB means")
    for key in variables_dict['target_types']: # loop through the target types
        print(f"----- target:{key} | probability:{variables_dict['target_type_probabilities'][key]} -----")
        for col in variables_dict['shared_attributes']: # loop through shared attributes
            if col != variables_dict['target_col_name']:
                continuous = False
                stdev = 'n/a'
                if col in variables_dict['continuous_columns']:
                    continuous = True
                    stdev = variables_dict[key][col]['stdev']
                print(f"key:{key} | col:{col} | continuous:{continuous} | mean:{variables_dict[key][col]['mean']} | stdev:{stdev}")
                if variables_dict['verbosity'] > 2:
                    for val in variables_dict[key][col]:
                        if val != 'mean':
                            print(f"target:{key} | col:{col} | val:{val} | p:{variables_dict[key][col][val]['probability']}")

# debugging info
#if variables_dict['verbosity'] > 0:
# print the train & test shapes (rows: subtract 1 for headers and 1 for starting from zero; cols: subtract 1 for "num" & 1 for "target")
print(f"train: {len(training_dict)-2} x {len(training_dict[0])-2} | test: {len(testing_dict)-2} x {len(testing_dict[0])-2} | shared: {len(variables_dict['shared_attributes'])-1}")

# debugging info
if variables_dict['verbosity'] > 2:
    # Print some of the rows from input files
    print(f"The first 2 training samples with target:{training_dict[0][variables_dict['target_col_index_train']]}:")
    for i in range(0, 2):
        print(f"\t{i} {training_dict[i]}")

    print(f"\nThe first 2 testing samples with target:{testing_dict[0][variables_dict['target_col_index_test']]}")
    for i in range(0, 2):
        print(f"\t{i} {testing_dict[i]}")

# loop through all testing data
for i in range(1, len(testing_dict) - 1):
#for i in range(1, 5):
    # create the test vector at the i-th row from only the shared test & train attributes
    test_vec = GetVectorOfOnlySharedAttributes(testing_dict, i, variables_dict)

    # set the k-nearest neighbors in the neighbors dict
    PopulateNearestNeighborsDicOfIndexes(training_dict, test_vec, variables_dict)

    # calculate and set the KNN predicted target and confidence in the variables dict
    AddKNNMajorityTypeToVarDict(variables_dict)

    # calculate and set the LDF predicted target and confidence in the variables dict
    AddCalcsOfPluginLDFToVarDic(test_vec, variables_dict)

    # calculate and set the NB predicted target and confidence in the variables dict
    AddTestProbabilityToVarDic(test_vec, variables_dict)
    
    # ----- Store the Confusion Matrix running counts -----
    # track KNN confusion matrix running totals
    TrackConfusionMatrixSums(testing_dict[i][variables_dict['target_col_index_test']], variables_dict['knn_best_target'], 'knn', variables_dict)
    # track LDF confusion matrix running totals
    TrackConfusionMatrixSums(testing_dict[i][variables_dict['target_col_index_test']], variables_dict['ldf_best_target'], 'ldf', variables_dict)
    # track NB confusion matrix running totals
    TrackConfusionMatrixSums(testing_dict[i][variables_dict['target_col_index_test']], variables_dict['nb_best_target'], 'nb', variables_dict)

    variables_dict['com_confidence'] = 0
    # loop through the classifier names to find best confidence level
    for key in variables_dict['classifiers']:
        if variables_dict['com_confidence'] < variables_dict[key + '_confidence']:
            variables_dict['com_confidence'] = variables_dict[key + '_confidence']
            variables_dict['com_best_target'] = variables_dict[key + '_best_target']

    # track Combined confusion matrix running totals
    TrackConfusionMatrixSums(testing_dict[i][variables_dict['target_col_index_test']], variables_dict['com_best_target'], 'com', variables_dict)
    # -----------------------------------------------------

    # add the running totals of predictions for KNN, LDF, & Combined
    AddRunningPredictionStatsToVarDict(testing_dict[i][variables_dict['target_col_index_test']], variables_dict)

    # reset kneighbors_dict
    InitNeighborsDict(variables_dict)

# print the three confusion matrices
PrintConfusionMatrix('knn', variables_dict)
PrintConfusionMatrix('ldf', variables_dict)
PrintConfusionMatrix('nb', variables_dict)
PrintConfusionMatrix('com', variables_dict)

# print the prediction stats for KNN, LDF, & Combined
print(f"\nall:      right |                  {variables_dict['com_knn_ldf_right']} \t| {round((variables_dict['com_knn_ldf_right']/variables_dict['test_runs_count']),2)}%")
print(f"com, knn: right | ldf:      wrong: {variables_dict['com_knn_right_ldf_wrong']} \t| {round((variables_dict['com_knn_right_ldf_wrong']/variables_dict['test_runs_count']),2)}%")
print(f"com, ldf: right | knn:      wrong: {variables_dict['com_ldf_right_knn_wrong']} \t| {round((variables_dict['com_ldf_right_knn_wrong']/variables_dict['test_runs_count']),2)}%")
print(f"knn:      right | com, ldf: wrong: {variables_dict['com_ldf_wrong_knn_right']} \t| {round((variables_dict['com_ldf_wrong_knn_right']/variables_dict['test_runs_count']),2)}%")
print(f"ldf:      right | com, knn: wrong: {variables_dict['com_knn_wrong_ldf_right']} \t| {round((variables_dict['com_knn_wrong_ldf_right']/variables_dict['test_runs_count']),2)}%")
print(f"                | all:      wrong: {variables_dict['com_knn_ldf_wrong']} \t| {round((variables_dict['com_knn_ldf_wrong']/variables_dict['test_runs_count']),2)}%")

# print LDF min & max values for reference
print(f"\nldf: min:{round(variables_dict['ldf_diff_min'],2)} | max:{round(variables_dict['ldf_diff_max'],2)}")

# debugging info - print LDF confidence == 0 summation
if variables_dict['ldf_confidence_zero'] > 0:
    print(f"ldf confidence <= 0: {variables_dict['ldf_confidence_zero']} \t| {round((variables_dict['ldf_confidence_zero']/variables_dict['test_runs_count']),2)}%")
