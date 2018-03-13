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
import sys
import time
matplotlib.use("TkAgg")
## LIBRARIES END
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
## THE TRAINNING MODEL FUNCTION
def testmodel(file_path='./CSV/test_raw.csv',log_line=False, log_time=False, folder_path='./model'):
    output_path = "./TestLog/"
    make_sure_path_exists(output_path)
    saved = sys.stdout
    timeStr = time.strftime("%Y%m%d-%H%M%S")
    fOut = open(str(output_path) + "/" + timeStr+".log", 'w')  # File where you need to keep the logs
    sys.stdout = writer(sys.stdout, fOut)

    print('\n \t TASK:  TEST THE MODEL')
    print(str(file_path))
    ## do something for retrain
    datasave = "./CSV/"
    # 1) Input data:
    if log_time == True:
        start = time.time()
    if file_path.endswith('.csv'):
        df_raw = pd.read_csv(file_path, error_bad_lines=False, low_memory=False)
        print("\n Data file loaded!")
    else:
        df_raw = parsing(input_data_path=file_path, output_path=datasave, errorlog_path=None, log_line=log_line)
    if 'IsCustomer' in df_raw.columns:
        df_raw = df_raw.drop("IsCustomer", 1)
        # df_raw = df_raw.reset_index(drop=True)

    print(" \t----------------------------------------------")
    if log_time == True:
        end = time.time()
        ParsingTime = (end - start)
        print(" >>>> Time: ({:04.2f} seconds) <<<<".format(ParsingTime))
    # 4) Cleaning:
    if log_time == True:
        start = time.time()
    df_clean = cleaning(df=df_raw, output_path=datasave, log_line=log_line)
    if log_time == True:
        end = time.time()
        CleaningTime = (end - start)
        print(" >>>> Time: ({:04.2f} seconds) <<<<".format(CleaningTime))
    del df_raw
    ### ======== Split markets ==========
    if log_time == True:
        start = time.time()

    if ('Time' in df_clean.columns):
        df_clean = df_clean.drop('Time',1)

    df_markets, n_markets = preparing(df=df_clean, log_line=log_line)
    if log_time == True:
        end = time.time()
        PreparingTime = (end - start)
        print(" >>>> Time: ({:04.2f} seconds) <<<<".format(PreparingTime))


    # 5) Apply model to each market
    print("\nTesting model ...")
    for market in df_markets:
        if log_time == True:
            startm = time.time()
        mkID = (market.MarketplaceId.unique())
        amzId = getAMZ(mkID)
        print('Market ID: ' + str(mkID) + '; amzID: ' + str(amzId))
        if (amzId != 'nothing'):
             # 5.1) Load the model + feature list
            print("\t Loading model ...")
            # folder_path = './model'
            getname = extractname(mkID, folder_path)
            for each in getname:
                file_base = os.path.basename(each)
                file_name = os.path.splitext(file_base)[0]
                file_ext = os.path.splitext(file_base)[1]
                if (getextension(each) == ".pkl"):  # if model load model
                    model_ = loadmodel(each)
                    model_time, model_market, model_clfs = file_name.split("_")
                    name_clf = classifierName(model_clfs)
                    print('\n \t Saved Time:' + str(model_time) + ' | Market: ' + str(model_market) + ' | Classifier: ' + str(name_clf) + '\n')
                    fealist_ = model_.fealist_
                # elif (getextension(each) == ".fea"):
                #         fealist_ = loadfeature(each)

                    market_,listmin_ = dataadd(market, amzId, test_ratio=0, listmin=model_.listmin_)
                    best_N_ = int(model_.n_features_)
                    market_ = datacut(market_, fealist_, best_N_)
                    ## split data
                    x_test, y_test = market_.iloc[:, 1:].values, market_.iloc[:, 0].values
                    # 5.2) Applying model
                    print("\t Applying model ...")
                    predictedclass, class_prob = applying(model_, x_test)
                    if (model_clfs == 4) or (model_clfs == 6):
                        predictedclass = [x if (x == 1) else -1 for x in predictedclass]
                    Pos_Probability = (class_prob[:, 1])
                    NewScore = market_["IsFulfilledByAmazon"] * Pos_Probability + Pos_Probability * market_["IdealPointCompetition"] + \
                               Pos_Probability * market_["IsFeaturedMerchant"]
                    ## From newscore to predictedclass.
                    market_["NewScore"] = NewScore
                    market_["CompetitionName"] = market["CompetitionName"]
                    market_["NewDecision"] = [0] * len(market_)
                    groups = market_.groupby(["CompetitionName"])
                    for name, group in groups:
                         gidx = group.index
                         groupmax = max(group.NewScore)
                         groupdec = ((group["NewScore"] == groupmax) - 0.5) * 2
                         market_['NewDecision'].iloc[gidx] = groupdec
                         # df_["max"].iloc[gidx] = [groupmax]*len(group)
                         del groupmax, groupdec

                    winner = (market_["NewDecision"].astype("int"))

                    precision2, recall2, fscore2, support2 = scoring(winner, y_test,True)
                    cnf_matrix2 = confusion_matrix(y_test, winner)
                    if (log_line == True):
                        print(cnf_matrix2)
                else:
                        print("Something wrong here!")

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
    sys.stdout = saved
    fOut.close()
## THE MAIN FUNCTION
import argparse, numpy as np

class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)

def main():
    clearscreen()
    # OLD VERSION
    parser = argparse.ArgumentParser(description='Amazon BuyBox model predictor 2.1 (Testing Procedure)')

    parser.add_argument('datapath', metavar='DATAPATH', type=str,
                        help='Testing XMLs folder')

    # parser.add_argument('-ratio', dest='testratio', metavar='Float[0,1)', type=restricted_float, default=0.1,
    #                     help='Ratio of test set in splitting training|testing data (Default: 0.1)')

    # parser.add_argument('-f1', dest='f1', metavar='True|False', type=restricted_bool, default=False,
    #                     help='Toggle to calculate classification results (Default: False)')

    parser.add_argument('-t', dest='time', metavar='True|False', type=restricted_bool, default=False,
                        help='Toggle to calculate execution time (Default: False)')

    parser.add_argument('-v', dest='log', metavar='True|False', type=restricted_bool, default=False,
                        help='Verbose (Default: False)')

    parser.add_argument('-modelpath', dest='modelpath', metavar='DATAPATH', type=str, default='./model',
                        help='The path of model folder')

    args = parser.parse_args()

    datapath = str(args.datapath)
    if datapath is '':
        datapath = './CSV/data_raw.csv'

    log = bool(args.log)
    time = bool(args.time)

    modelpath = str(args.modelpath)
    print(['datapath= ' + str(datapath) + '\n -v= ' + str(log) + '\n -t= ' + str(time) + '\n -modelpath= ' + str(modelpath)])
    testmodel(file_path=datapath, log_line=log, log_time=time, folder_path=modelpath)
if __name__ == "__main__":
    main()
