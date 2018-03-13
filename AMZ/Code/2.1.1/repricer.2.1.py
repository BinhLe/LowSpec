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
import warnings
import numpy as np
import pandas as pd
import os
from scipy.stats import norm, multivariate_normal
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

import matplotlib
matplotlib.use("TkAgg")
## LIBRARIES END
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
## SOME DEFINITIONS
def extractname(mkID, file_path, clfs = 1):
    if (clfs == 7):
        out_name = []
        for name in glob.glob(file_path + "/*_" + str(mkID[0]) + '*.pkl'):
            out_name.append(name)
    else:
        out_name = []
        for name in glob.glob(file_path + "/*_" + str(mkID[0]) + '_' + str(clfs) +'.pkl'):
            out_name.append(name)

    # print("\t The feature list and model: \n \t" + str(out_name))
    return out_name

def repricing(file_path='None',log_line=False,modelpath='.\model',clfs = 1, file_list=pd.DataFrame([])): # fild_path is dataframe

    ## setup the menu for the program
    # if(not file_path.endswith('.xml') or  file_path == 'None'):
    if file_path.empty:
        print('Wrong input')
        return False
    else:
        # 1) Load the data & XML

        print("\n =================================================================================")
        print(' REPRICING FILE: ' + str(file_path['CompetitionName'].unique()[0]))
        print(" =================================================================================")
        # if file_path.endswith('.csv'):
        #     df_file = pd.read_csv(file_path, error_bad_lines=False, low_memory=False)
        # else:
        #     df_file = parsing_file(input_data_path=file_path, log_line=log_line)

        df_file = file_path.copy()
        if (df_file.empty):
            print('File ' + str(file_path) + ' is wrong in format! Exit!')
            return False

    # XMLname = os.path.basename(file_path)
    # XMLname = os.path.splitext(XMLname)[0]
        XMLname = (df_file['CompetitionName'].unique())[0]
        if log_line==True:
            print("\n \t Auction data loaded!")
            print(" \t----------------------------------------------")
        # 2.1) load data and related data
        data_path = "./CSV/data_raw.csv"
        df_raw = pd.read_csv(data_path, error_bad_lines=False, low_memory=False)
        if log_line==True:
            print("\n\t Historical data loaded!")
            print(" \t---------------------------------------------- \n")
        mkID = (df_file.MarketplaceId.unique())
        ## get productID
        prodId = df_file.ProductId.unique()[0]
        df_customer = df_file.loc[df_file.IsCustomer == 1]
        df_notcustomer = df_file.loc[df_file.IsCustomer != 1]

        if (len(df_customer)==0):
            print("\n There is no customer in XML !!! Exit !")
            return False
        elif (len(df_customer) > 1):
            print("\n There is more than one customer in XML !!! Exit !")
            return False

        amzId = getAMZ(mkID)

        # Prepare data
        # 3) Set up Bins
        ## get customer
        customer_id = df_customer.iloc[0].SellerId
        customer_price = df_customer.iloc[0].ListingPrice + df_customer.iloc[0].ShippingPrice
        customer_status = df_customer.iloc[0].IsBuyBoxWinner
        customer_time = df_customer.iloc[0].ShippingTime_maxHours
        flag = False
        ## get winner
        # winner_id = df_file.loc[df_file.IsBuyBoxWinner == 1].SellerId
        if len(df_file.loc[df_file.IsBuyBoxWinner == 1]) == 0:
            print("There is no Winner in this XML, get the minimum price !!!")
            winner_price = min(df_file.ListingPrice + df_file.ShippingPrice)
            winner_time = min(
                df_file['ShippingTime_maxHours'].loc[df_file.ListingPrice + df_file.ShippingPrice == winner_price])
            flag = True
        elif len(df_file.loc[df_file.IsBuyBoxWinner == 1]) > 1:
            print("There is more than one Winner in this XML, get the Winner with minimum price !!!")
            winner_price = min(df_file.loc[df_file.IsBuyBoxWinner == 1].ListingPrice + df_file.loc[
                df_file.IsBuyBoxWinner == 1].ShippingPrice)
            winner_time = min(
                df_file['ShippingTime_maxHours'].loc[df_file.ListingPrice + df_file.ShippingPrice == winner_price])
        else:
            winner_price = min(df_file.loc[df_file.IsBuyBoxWinner == 1].ListingPrice + df_file.loc[
                df_file.IsBuyBoxWinner == 1].ShippingPrice)
            winner_time = min(
                df_file['ShippingTime_maxHours'].loc[df_file.ListingPrice + df_file.ShippingPrice == winner_price])

        # consider_price = np.float(winner_price - np.exp((winner_time - customer_time)/24))
        # print(np.exp((winner_time - customer_time)/24))
        # winner_price = consider_price
        # get the ten percentage threshold of winner price
        ten_percent_win_up = np.float(winner_price + winner_price * 0.1)
        ten_percent_win_down = np.float(winner_price - winner_price * 0.1)

        ten_percent_cus_up = np.float(customer_price + customer_price * 0.1)
        ten_percent_cus_down = np.float(customer_price - customer_price * 0.1)

        ## get the min and max threshold
        min_threshold = np.float(min(df_file.ListingPrice + df_file.ShippingPrice))
        # max_theshold = max(df_file.ListingPrice + df_file.ShippingPrice)

        ## get the difference from winner to customer price
        # diff_price =  customer_price - winner_price

        # maxp = min(customer_price,ten_percent_win_up)
        # minp = max(ten_percent_win_down,min_threshold)
        # midp = winner_price

        lowerbound, upperbound, middlepoint = getboundary(min_threshold, winner_price, ten_percent_win_up,
                                                          ten_percent_win_down, customer_price, ten_percent_cus_up,
                                                          ten_percent_cus_down, customer_status)

        final_list = create_bins(minp=lowerbound, midp=middlepoint, maxp=upperbound, fplot=False)
        # weight = multivariate_normal.pdf(final_list, mean=middlepoint)


        ## not customer
        # not_customer_price = df_notcustomer.ListingPrice + df_notcustomer.ShippingPrice
        ## retrieve product by ID
        pool_prod = df_raw.loc[df_raw.ProductId == prodId]
        Prod_LandedPrice = pool_prod.ListingPrice + pool_prod.ShippingPrice
        if (len(Prod_LandedPrice) != 0):
            min_product_price = min(Prod_LandedPrice)
        else:
            min_product_price = min_threshold

        ## get probabbilities of XML competition
        df_file2 = pd.concat([df_customer, df_notcustomer])
        df_file2["LandedPrice"] = df_file2.ListingPrice + df_file2.ShippingPrice
        df_file2["DifftoMinLandedPriceProduct"] = (df_file2.ListingPrice + df_file2.ShippingPrice) - min_product_price

        T = len(final_list)
        ## duplicate with different prices
        df_candidates = pd.concat([df_customer] * T, ignore_index=True)  # Ignores the index
        # add the suitable features
        df_candidates["LandedPrice"] = final_list
        # pool_prod["LandedPrice"] = Prod_LandedPrice
        # df_notcustomer["LandedPrice"] = not_customer_price

        df_candidates["DifftoMinLandedPriceProduct"] = final_list - min_product_price
        # pool_prod["DifftoMinLandedPriceProduct"] = Prod_LandedPrice - min_product_price
        # df_notcustomer["DifftoMinLandedPriceProduct"] = not_customer_price - min_product_price
        df_final_data = df_candidates.append(df_file2, ignore_index=True)

        # 2) Load the model + feature list
        folder_path = modelpath
        getname = extractname(mkID, folder_path, clfs=clfs)
        idx = 0
        reprices = pd.DataFrame([])

        for each in getname:
            file_base = os.path.basename(each)
            file_name = os.path.splitext(file_base)[0]
            file_ext = os.path.splitext(file_base)[1]

            if (getextension(each) == ".pkl"):  # if model load model
                model = loadmodel(each)
                model_time, model_market, model_clfs = file_name.split("_")
                name_clf = classifierName(model_clfs)
                featlist = model.fealist_
                if log_line == True:
                    print('\n \t Saved Time:' + str(model_time) + ' | Market: ' + str(model_market) + ' | Classifier: ' + str(
                        name_clf) + '\n')
            else:
                print("Something wrong here!")
                return False

            #

            # df_tmp = df_final_data.copy()
            df_final_data_managed,listmin_ = datamanage(df_final_data, 0, featlist, model.n_features_, amzId,listmin=model.listmin_)
            weight = multivariate_normal.pdf(df_final_data["LandedPrice"].values, mean=winner_price)

            df_tmp = df_final_data_managed.copy()

            if "IsFulfilledByAmazon" not in df_tmp.columns:
                df_tmp["IsFulfilledByAmazon"] = df_final_data["IsFulfilledByAmazon"]
            if "IsFeaturedMerchant" not in df_tmp.columns:
                df_tmp["IsFeaturedMerchant"] = df_final_data["IsFeaturedMerchant"]
            if 'IsBuyBoxWinner' in df_tmp.columns:
                df_tmp["IsBuyBoxWinner"] = df_final_data["IsBuyBoxWinner"]
            if 'IdealPointCompetition' in df_tmp.columns:
                df_tmp["IdealPointCompetition"] = df_final_data_managed["IdealPointCompetition"]
            if 'CompetitionName' not in df_tmp.columns:
                df_tmp["CompetitionName"] = df_final_data["CompetitionName"]

            if 'IsBuyBoxWinner' in df_final_data_managed.columns:
                df_final_data_managed = df_final_data_managed.drop(['IsBuyBoxWinner'], 1)

            df_final_data_managed.reset_index(drop=True)
            x_test = df_final_data_managed.iloc[:, 0:].values
            predictedclass, class_prob = applying(model, x_test)
            if (model_clfs == 4) or (model_clfs == 6):
                predictedclass = [x if (x == 1) else -1 for x in predictedclass]
            Pos_Probability = (class_prob[:, 1])

            df_tmp["prob"] = Pos_Probability
            df_tmp = decisionmaking(df_tmp)
            # printresults(data=df_tmp)
            final_predictedclass = (df_tmp["NewDecision"].head(len(df_candidates)))
            #new
            df_tmp["NewScore"] = minmaxscalerb(minmaxscalerb(df_tmp["NewScore"]) + minmaxscalerb(weight))
            # df_tmp["NewScore"] = (minmaxscalerb(df_tmp["NewScore"]))
            if len(df_candidates) == 1:
                final_class_prob = df_tmp["NewScore"][0]
                final_class_prob = final_class_prob * 100
                final_class_prob = [final_class_prob]
            else:
                # print(df_tmp[["IsBuyBoxWinner","LandedPrice",'ShippingTime_maxHours',"NewScore",'IsFeaturedMerchant','IsFulfilledByAmazon','SellerFeedbackRating','SellerFeedbackCount', 'CompeteAmzCompetition']])
                final_class_prob = df_tmp["NewScore"].head(len(df_candidates)).reindex()
                final_class_prob *= 100

            df_auction = df_tmp.loc[len(df_candidates):].copy()
            if(sum((df_auction["IsBuyBoxWinner"]==1)*1)==0):
                winner_score = 0
            elif sum((df_auction["IsBuyBoxWinner"]==1)*1)>1:
                winner_score = np.around(
                    (df_auction["NewScore"].loc[df_auction["IsBuyBoxWinner"] == 1].values) * 100, decimals=3)

            else:
                winner_score = np.around(np.float(df_auction["NewScore"].loc[df_auction["IsBuyBoxWinner"]==1].values)*100, decimals=3)
            df_notcustomer["NewScore"] = (df_auction["NewScore"].loc[len(df_candidates)+1:].values)*100

            customer_score = np.float(df_auction["NewScore"].head(1))*100
            ## Fitting
            z = np.polyfit(final_list, (final_class_prob), 5)
            f = np.poly1d(z)
            final_class_prob = f(final_list)
            final_class_prob[final_class_prob<0] =0
            final_class_prob[final_class_prob > 100] = 100

             # 6) Get the available solution
            df_result2  = result_display(customer_id, (customer_price), customer_status, final_list, final_class_prob, df_notcustomer,
                           np.float(winner_price), plotforus=False, flag=flag, nameXML= XMLname, winner_score=winner_score, customer_score=customer_score, name_clf= name_clf, log_line=log_line)
            reprices['File'] = (XMLname)
            reprices['CurrentPrice'] = customer_price
            reprices['WinnerPrice'] = winner_price
            if (not file_list.empty):
                reprices['NextFile'] = file_list.loc[file_list.File == XMLname, 'NextId']
                reprices['NextPrice'] = file_list.loc[file_list.File == XMLname, 'NextPrice']

            reprices[name_clf] = df_result2["Price"].head(1).values

        # print('\n=========== Summarization ==========')
        # print(reprices)

        return(reprices)
def select_reprice_item(df_raw):
    df_raw['Time'] = np.log(df_raw['Time'])
    FindingTable = df_raw.loc[
        df_raw.IsBuyBoxWinner == 1, ['CompetitionName', 'SellerId', 'Time', 'MarketplaceId', 'ProductId','IsCustomer', 'ListingPrice',
                                   'ShippingPrice']]
    FindingTable.columns = ['File', 'WinnerId', 'Time', 'MarketplaceId', 'ProductId', 'IsCustomer', 'ListingPrice', 'ShippingPrice']
    FindingTable['LandedPrice'] = FindingTable['ListingPrice'] + FindingTable['ShippingPrice']
    FindingTable = FindingTable.drop('ListingPrice', 1)
    FindingTable = FindingTable.drop('ShippingPrice', 1)
    result = FindingTable.sort_values(['Time'])
    result = result.reset_index(drop=True)

    final_table = result
    final_table['Next'] = result['IsCustomer']
    final_table['NextId'] = result['File']
    final_table['NextPrice'] = result['LandedPrice']
    final_table[['File', 'WinnerId', 'Time', 'MarketplaceId', 'ProductId','IsCustomer', 'LandedPrice']] = \
    final_table.groupby(['MarketplaceId','WinnerId','ProductId'])[
        'File', 'WinnerId', 'Time', 'MarketplaceId','ProductId', 'IsCustomer', 'LandedPrice'].transform(lambda x: x.shift(1))
    final_table.drop(final_table.index[:1], inplace=True)
    final_table = final_table.dropna()

    file_list = pd.DataFrame([], ['File', 'CurrentPrice', 'NextFile', 'NextWinnerPrice'])
    groups = final_table.groupby(['MarketplaceId','WinnerId','ProductId'])
    for name, group in groups:
        tmp = group.query('IsCustomer == 0 and Next == 1')
        if not tmp.empty:
            file_list = file_list.append(tmp[['File', 'LandedPrice', 'NextId', 'NextPrice']], ignore_index=True)
    file_list = file_list.dropna()
    file_list.to_csv('reprice_list.txt', sep=',', index=False)
    df_out = df_raw[df_raw['CompetitionName'].isin(list(file_list['File']))]
    return df_out, file_list

def repricing_multi(file_path=None,log_line=False,modelpath = None, clfs=1):
    print('\n \t TASK: RECOMMEND PRICE')
    # check the folder at file_path
    if (modelpath == None):
        modelpath = './model'

    if (file_path is None):
        raise NameError('Data path is not declared! Cannot run!')
    elif file_path.endswith('.xml'):
        df_file = parsing_file(input_data_path=file_path, log_line=log_line)
        reprices = repricing(file_path=df_file, log_line=log_line, modelpath=modelpath, clfs = clfs)
        print(type(reprices).__name__)
        if type(reprices).__name__ in ['bool']:
            df_result = pd.DataFrame([])
        else:
            df_result = reprices.copy()
    elif file_path.endswith('.csv'):
        df_file = pd.read_csv(file_path, error_bad_lines=False, low_memory=False)
        df_file, file_list = select_reprice_item(df_file)
        if (df_file.empty):
            print('No data for repricing! Exit!')
            exit(True)
        else:
            n_auction = len(df_file['CompetitionName'].unique())
            if (n_auction == 1):
                reprices = repricing(file_path=df_file, log_line=log_line, modelpath=modelpath, clfs = clfs)
                if type(reprices).__name__ in ['bool']:
                    df_result = pd.DataFrame([])
                else:
                    df_result = reprices.copy()
            else:

                df_result = pd.DataFrame([])
                groups = df_file.groupby(["CompetitionName"])
                i = 0
                n__ = len(groups)
                for name, group in groups:
                    # gidx = group.index
                    # print('================ REPRICE THE AUCTION: ' + name + ' ================')
                    i = i + 1
                    print('File: ' + str(i) + '/' + str(n__))
                    reprices = repricing(file_path=group, log_line=log_line, modelpath=modelpath, clfs=clfs,file_list=file_list)
                    if not type(reprices).__name__ in ['bool']:
                        df_result = df_result.append(reprices, ignore_index=True)
                    df_result.to_csv('reprice_Results.txt', sep=',', index=False)

    elif glob.glob(file_path + '\*.xml'):
        full_file_paths = get_filepaths(file_path)
        # n = len(full_file_paths)
        xml_n = [f for f in full_file_paths if f.endswith('.xml')]
        if (len(xml_n) == 0):
            print("No xml file in this folder! Break!")
            exit(True)
        # fileindex = 0
        df_result = pd.DataFrame([])
        # for each file in folder, do repricer file
        for f in (full_file_paths):
            df_file = parsing_file(input_data_path=f, log_line=log_line)
            reprices = repricing(file_path=df_file, log_line=log_line,modelpath=modelpath, clfs = clfs)
            if not type(reprices).__name__ in ['bool']:
                df_result = df_result.append(reprices, ignore_index=True)
            df_result.to_csv('reprice_Results.txt', sep=',', index=False)
    else:
        print('No data for repricing! Exit!')
        exit(True)
    print('\n=========== OVER ALL ==========')
    print(df_result)
    df_result.to_csv('reprice_Results.txt', sep=',', index=False)
    return df_result

## MAIN FUNCTION
def main():
    clearscreen()
    # OLD VERSION
    parser = argparse.ArgumentParser(description='Amazon Repricer 2.0 ...')

    parser.add_argument('datapath', metavar='DATAPATH', type=str,
                        help='Path to XML file')

    # parser.add_argument('-cl', dest='clf', metavar='1|2|3|4|5|6', type=restricted_int, default=1,
    #                     help='Select the classifer: 1:RandomForest 2:LogisticRegression 3:KNeighbors 4:SVM-Rbf 5: AdaBoost 6: XGBoost 7:All (Default:1) ')
    parser.add_argument('-cl', dest='clf', metavar='1|2|3|4|5|6', action=Store_as_array, type=int, nargs='+', default=1,
                        help='Select the classifer: 1:RandomForest 2:LogisticRegression 3:KNeighbors 4:SVM-Rbf 5: AdaBoost 6: XGBoost 7:All (Default:1) ')

    parser.add_argument('-modelpath', dest='modelpath', metavar='DATAPATH', type=str, default='./model',
                        help='The path of model folder')

    parser.add_argument('-v', dest='log', metavar='True|False', type=restricted_bool, default=True,
                        help='Verbose (Default: False)')

    args = parser.parse_args()

    datapath = str(args.datapath)
    if datapath is '':
        datapath = 'None'

    logs = (args.log)
    modelpath = str(args.modelpath)
    # classifier = int(args.clf)
    # name = classifierName(classifier)
    classifier = (args.clf)
    ## get name of classifier
    # if len(classifier) > 1:
    if (type(classifier).__name__ in ('list', 'tuple', 'ndarray')):
        classifier = list(classifier)
        name = []
        for cl in classifier:
            name = name + [str(classifierName(cl))]
    else:
        classifier = int(classifier)
        name = classifierName(classifier)

    print('datapath= ' + str(datapath) +  '\n -modelpath= ' + str(modelpath) + '\n -cl= ' + str(name) + '\n -v= ' + str(logs))

    repricing_multi(file_path=datapath, log_line=logs, modelpath = modelpath, clfs=classifier)
##
if __name__ == "__main__":
    main()
