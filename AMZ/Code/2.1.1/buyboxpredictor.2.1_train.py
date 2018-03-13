#!/usr/bin/python
"""
=============================================================================================================
                                    Python interface of ML4AMZ project
                                            Version: 2.1
=============================================================================================================
This software was developed within project IP/2016/0436 by researchers at University College Dublin (UCD).

UCD hereby grants the recipient a personal, non-transferable, non-exclusive license to use the software on
the recipient's computer system during the trial period, solely for the purposes of the evaluation of the
software by the recipient, for use in the recipient's business.
=============================================================================================================
"""

from repricer21libs import *
import argparse
import time
import warnings
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
## LIBRARIES END
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
## THE TRAINNING MODEL FUNCTION
def trainmodel(file_path='./CSV/data_raw.csv',log_line=False, log_time=False, clfs=1):
    print('\n \t TASK:  TRAIN THE MODEL')
    ## do something for retrain
    datasave = "./CSV/"
    # 1) Input data:
    print("Parsing ...")
    if log_time == True:
        start = time.time()
    if file_path.endswith('.csv'):
        df_raw = pd.read_csv(file_path, error_bad_lines=False, low_memory=False)
        print("\n Data file loaded!")
    else:
        df_raw = parsing(input_data_path=file_path, output_path=datasave, errorlog_path=None, log_line=log_line)

        # df_raw = df_raw.reset_index(drop=True)

    print(" \t----------------------------------------------")
    if log_time == True:
        end = time.time()
        ParsingTime = (end - start)
        print(" >>>> Time: ({:04.2f} seconds) <<<<".format(ParsingTime))
    # 4) Cleaning:
    if log_time == True:
        start = time.time()
    print("Cleaning ...")
    df_clean = cleaning(df=df_raw, output_path=datasave, log_line=log_line)
    if log_time == True:
        end = time.time()
        CleaningTime = (end - start)
        print(" >>>> Time: ({:04.2f} seconds) <<<<".format(CleaningTime))
    del df_raw
    ### ======== Split markets ==========
    if log_time == True:
        start = time.time()
    # query = input("Do you have .CSV markets file? [Y/N]: ")
    # if (query == "N" or query == "n"):
    if 'Time' in df_clean.columns:
        df_clean = df_clean.drop("Time",1)
    if 'IsCustomer' in df_clean.columns:
        df_clean = df_clean.drop("IsCustomer", 1)

    df_markets, n_markets = preparing(df=df_clean, log_line=log_line)
    print('Data saved!')
    if log_time == True:
        end = time.time()
        PreparingTime = (end - start)
        print(" >>>> Time: ({:04.2f} seconds) <<<<".format(PreparingTime))
    del df_clean

    # 5) F.I. computing &  model:
    print(" =============== Model Construction ====================")
    if log_time == True:
        start = time.time()

    ## Run for all markets
    model_path = "./model/"
    make_sure_path_exists(model_path)
    print("Model is stored in: " + model_path)
    # test_ratio = input("Please select the test ratio splitting (press 'Enter' to get default 0.3): ")
    # test_ratio = 0.3

    for market in df_markets:
        if log_time == True:
            startm = time.time()
        mkID = ''.join(market.MarketplaceId.unique())
        amzId = getAMZ(mkID)
        # df_ = addAMZfeatures(df_, amzId, norm=False)

        print('Market ID: ' + str(mkID) + '; amzID: ' + str(amzId))
        if (amzId != 'nothing'):
            ## Check Feature Importance and get best cut
            market_, listmin_ = dataadd(market, amzId, test_ratio=0, listmin={})
            # print([len(market),len(market_)])
            fealist_ = computeFI(market_, 10, bootstrap=True, plot=False)
            best_N_ = len(market_.columns)
            market_ = datacut(market_, fealist_, best_N_)
            market_.to_csv("./CSV/Market_csv/" + str(mkID) + "_train.csv", sep=',', index=False)

            x_train, y_train = market_.iloc[:, 1:].values, market_.iloc[:, 0].values
            if type(clfs).__name__ in ('list', 'tuple', 'ndarray'):
                clfs_list = list(clfs)
                for cl in clfs_list:
                    name = classifierName(cl)
                    print('\n \t Classifier Training : ' + str(name))
                    if (cl == 4) or (cl == 6):
                        y_train = [x if (x == 1) else 0 for x in y_train]
                    model_ = training(x_train, y_train, cl)  # TRAIN STEP
                    model_.listmin_ = listmin_
                    model_.n_features_ = best_N_
                    model_.clfs = cl
                    model_.fealist_ = fealist_
                    ## save model after training
                    savemodel(model_, model_path, mkID, cl)
                    savefeaturelist(fealist_, model_path, mkID, cl)
            else:
                clfs = int(clfs)
                if clfs == 7:
                    clfs_list = [1,2,3,5,6,4]
                    for cl in clfs_list:
                        name = classifierName(cl)
                        print('\n \t Classifier Training : '+ str(name))
                        if (cl == 4) or (cl == 6):
                            y_train = [x if (x == 1) else 0 for x in y_train]
                        model_ = training(x_train, y_train, cl)  # TRAIN STEP
                        model_.listmin_ = listmin_
                        model_.n_features_ = best_N_
                        model_.clfs = cl
                        model_.fealist_ = fealist_
                        ## save model after training
                        savemodel(model_, model_path, mkID, cl)
                        savefeaturelist(fealist_, model_path, mkID, cl)
                else:
                    if (clfs == 4) or (clfs == 6):
                        y_train = [x if (x == 1) else 0 for x in y_train]
                    model_ = training(x_train, y_train, clfs)   # TRAIN STEP
                    model_.listmin_ = listmin_
                    model_.n_features_ = best_N_
                    model_.clfs = clfs
                    model_.fealist_ = fealist_
                    ## save model after training
                    savemodel(model_, model_path, mkID, clfs)
                    savefeaturelist(fealist_, model_path, mkID,clfs)

            print("Model for market " + mkID + " is saved!")

        else:
            print('Cannot find marketplace ID')

        if log_time == True:
            endm = time.time()
            ComputingTimeM = (endm - startm)
            print(" >>>> Time (market: " + str(mkID) + "): ({:04.2f} seconds) <<<<".format(ComputingTimeM))

    if log_time == True:
        end = time.time()
        ComputingTime = (end - start)
        print(" >>>> Time: ({:04.2f} seconds) <<<<".format(ComputingTime))

        print(" \t----------------------------------------------")
        print(" >>>> Total time: ({:04.2f} seconds) <<<<".format(
            ParsingTime + CleaningTime + PreparingTime + ComputingTime))
        print(" \t----------------------------------------------")
    print(" =============== Model saving end ====================")
## THE MAIN FUNCTION
def main():
    clearscreen()
    parser = argparse.ArgumentParser(description='Amazon BuyBox model predictor 2.1 (Training Procedure)')

    parser.add_argument('datapath', metavar='DATAPATH', type=str,
                        help='Training XMLs folder')

    # parser.add_argument('-ratio', dest='testratio', metavar='Float[0,1)', type=restricted_float, default=0.1,
    #                     help='Ratio of test set in splitting training|testing data (Default: 0.1)')

    # parser.add_argument('-f1', dest='f1', metavar='True|False', type=restricted_bool, default=False,
    #                     help='Toggle to calculate classification results (Default: False)')

    # parser.add_argument('-cl', dest='clf', metavar='1|2|3|4|5|6', type=restricted_int, default=1,
                        # help='Select the classifer: 1:RandomForest 2:LogisticRegression 3:KNeighbors 4:SVM-Rbf 5: AdaBoost 6: XGBoost 7:All (Default:1) ')

    parser.add_argument('-t', dest='time', metavar='True|False', type=restricted_bool, default=False,
                        help='Toggle to calculate execution time (Default: False)')

    parser.add_argument('-v', dest='log', metavar='True|False', type=restricted_bool, default=False,
                        help='Verbose (Default: False)')
    parser.add_argument('-cl', dest='clf',metavar='1|2|3|4|5|6',action=Store_as_array, type=int, nargs='+', default=1,
                        help='Select the classifer: 1:RandomForest 2:LogisticRegression 3:KNeighbors 4:SVM-Rbf 5: AdaBoost 6: XGBoost 7:All (Default:1) ')
    # ^ specify as the action

    args = parser.parse_args()

    datapath = str(args.datapath)
    if datapath is '':
        datapath = './CSV/data_raw.csv'

    log = bool(args.log)
    time = bool(args.time)
    classifier = (args.clf)

    ## get name of classifier
    if type(classifier).__name__ in ('list', 'tuple', 'ndarray'):
        classifier = list(classifier)
        name = []
        for cl in classifier:
            name = name + [str(classifierName(cl))]
    else:
        classifier = int(classifier)
        name = classifierName(classifier)
    print(['datapath= ' + str(datapath), '-v= ' + str(log), '-t= ' + str(time), '-cl= ' + str(name)])
    trainmodel(file_path=datapath, log_line=log, log_time=time, clfs=classifier)

if __name__ == "__main__":
    main()

