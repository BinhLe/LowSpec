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

import argparse
import errno
import glob
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from sys import platform as _platform

import numpy as np
import pandas as pd
from lxml import etree
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier

# import time
# import tkinter as Tkinter
# from tkinter import filedialog as tkFileDialog
## LIBRARIES END
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

## FUNCTION TO CHECK DUPLICATION IN COLUMNS
def duplicate_columns(frame):
    df = frame.copy()
    if 'MarketplaceId' in df.columns:
        df = df.drop(['MarketplaceId'], 1)
    if 'ProductId' in df.columns:
        df = df.drop(['ProductId'], 1)
    if 'SellerId' in df.columns:
        df = df.drop(['SellerId'], 1)
    if 'CompetitionName' in df.columns:
        df = df.drop(['CompetitionName'], 1)

    groups = df.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:, i].values
            for j in range(i + 1, lcs):
                ja = vs.iloc[:, j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break

    return dups
# FUNCTION TO CREATE THE PATH
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
# FUNCTION TO GET LIST OF FILENAME IN DIRECTORY
def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

## FUNCTION TO PARSE DATA FROM XML
def parsing_file(input_data_path=None, log_line=False):
    if (log_line == True):
        print('Parsing ...')
    if (input_data_path is None):
        raise NameError('Data path is not declared! Cannot run!')

    header = ["IsBuyBoxWinner",
              "MarketplaceId",
              "ProductId",
              "SellerId",
              "ConditionNotes",
              "IsFeaturedMerchant",
              "IsFulfilledByAmazon",
              "ListingPrice",
              "ShippingPrice",
              "ShippingTime_maxHours", "ShippingTime_minHours", "ShippingTime_availtype",
              "ShipsDomestically",
              "ShipsFromCountry", "ShipsFromState",
              "SellerFeedbackRating", "SellerFeedbackCount",
              "SubCondition",
              "IsCustomer",
              "Time",
              "CompetitionName"]
    df_out = pd.DataFrame([], columns=header)

    f = open(input_data_path, "r")
    XML_name = os.path.basename(input_data_path)
    XML_name = os.path.splitext(XML_name)[0]

    tree = etree.parse(f)
    root = tree.getroot()
    for tags in (root.iter("NotificationMetaData")):
        find_node = tags.find("SellerId")
        if (find_node is not None):
            CustomerId = str((find_node.text))
        else:
            CustomerId = ""

    for tags in (root.iter("OfferChangeTrigger")):
        # print(tags)
        find_node = tags.find("MarketplaceId")
        if (find_node is not None):
            MarketplaceId = str((find_node.text))
        else:
            MarketplaceId = ""

        find_node = tags.find("ASIN")
        if (find_node is not None):
            ProductId = str((find_node.text))
        else:
            ProductId = ""

        find_node = tags.find("TimeOfOfferChange")
        if (find_node is not None):
            TimeOffer = str((find_node.text))
        else:
            TimeOffer = ""
        Time = datetime.strptime(TimeOffer, '%Y-%m-%dT%H:%M:%S.%fZ')
        TD = (Time - datetime(2016, 1, 1)).total_seconds()
        # year = int(dt.year)
        # month = int(dt.month)
        # day = int(dt.day)

    for tags in (root.iter("Offer")):
        # print(tags)
        find_node = tags.find("IsBuyBoxWinner")
        if find_node.text == "true" or find_node.text == "True":
            IsBuyBoxWinner = 1
        else:
            IsBuyBoxWinner = -1

        find_node = tags.find("SellerId")
        if find_node is not None:
            SellerId = str((find_node.text))
        else:
            SellerId = ""

        if (SellerId == CustomerId):
            IsCustomer = 1
        else:
            IsCustomer = 0

        find_node = tags.find("ConditionNotes")
        if (find_node is None):
            ConditionNotes = 0
        else:
            ConditionNotes = 1

        find_node = tags.find("IsFeaturedMerchant")
        if find_node.text == "true" or find_node.text == "True":
            IsFeaturedMerchant = 1
        else:
            IsFeaturedMerchant = 0

        find_node = tags.find("IsFulfilledByAmazon")
        if find_node.text == "true" or find_node.text == "True":
            IsFulfilledByAmazon = 1
        else:
            IsFulfilledByAmazon = 0

        find_node = tags.find("ListingPrice")
        if (find_node is not None):
            ListingPrice = float(find_node.findtext('Amount', default='None'))

        else:
            ListingPrice = -1

        find_node = tags.find("Shipping")
        if (find_node is not None):
            ShippingPrice = float(find_node.findtext('Amount', default='None'))
        else:
            ShippingPrice = "-1"

        find_node = tags.find("ShippingTime")
        if (find_node is not None):
            minHours = int(find_node.attrib["minimumHours"])
            maxHours = int(find_node.attrib["maximumHours"])
            availType = str(find_node.attrib["availabilityType"])
        else:
            minHours = -1
            maxHours = -1
            availType = ""

        find_node = tags.find("ShipsDomestically")
        if find_node.text == "true" or find_node.text == "True":
            ShipsDomestically = 1
        else:
            ShipsDomestically = 0

        find_node = tags.find("ShipsFrom")
        if (find_node is not None):
            ShipsFromCountry = str(find_node.findtext('Country', default='None'))
            ShipsFromState = str(find_node.findtext('State', default='None'))

        else:
            ShipsFromCountry = ""
            ShipsFromState = ""

        find_node = tags.find("SellerFeedbackRating")
        if (find_node is not None):
            SellerFeedbackRating = int(find_node.findtext('SellerPositiveFeedbackRating', default='None'))

            SellerFeedbackCount = int(find_node.findtext('FeedbackCount', default='None'))
        else:
            SellerFeedbackRating = -1
            SellerFeedbackCount = -1

        find_node = tags.find("SubCondition")
        if (find_node is not None):
            SubCondition = str((find_node.text))
        else:
            SubCondition = ""
        rowstr = [IsBuyBoxWinner,
                  MarketplaceId,
                  ProductId,
                  SellerId,
                  ConditionNotes,
                  IsFeaturedMerchant,
                  IsFulfilledByAmazon,
                  ListingPrice,
                  ShippingPrice,
                  maxHours, minHours,
                  availType,
                  ShipsDomestically,
                  ShipsFromCountry,
                  ShipsFromState,
                  SellerFeedbackRating,
                  SellerFeedbackCount,
                  SubCondition,
                  IsCustomer,
                  TD,
                  XML_name]
        row = dict(zip(header, rowstr))
        # row_s = pd.Series(row)
        df_out = df_out.append(row, ignore_index=True)
        # df_out[['IsBuyBoxWinner', 'ListingPrice','ShippingPrice','maxHours','minHours','year','month','day']] = \
        #     df[['IsBuyBoxWinner', 'ListingPrice','ShippingPrice','maxHours','minHours','year','month','day']].apply(pd.to_numeric,errors='ignore')

        df_out = df_out.convert_objects(convert_dates=True, convert_numeric=True, convert_timedeltas=True, copy=True)
        # print(df_out["Time"])
        # df_out["Time"] = handleTime(df_out["Time"])
        # print(df_out)
    return df_out

def parsing(input_data_path=None, output_path=None, errorlog_path=None, log_line=None):
    """
    ============= Parsing =============
    """
    if (log_line == True):
        print(parsing.__doc__)
    if (input_data_path is None):
        raise NameError('Data path is not declared! Cannot run!')
    # ----
    if (output_path is None):
        output_path = "./CSV/"
    else:
        make_sure_path_exists(output_path)
    output_path = output_path + 'data_raw.csv'
    # ----
    if (errorlog_path is None):
        errorlog_path = "./"
    else:
        make_sure_path_exists(errorlog_path)
    errorlog_path = errorlog_path + 'error_log.txt'

    if (log_line is None):
        log_line = True

    error_file = open(errorlog_path, mode="w")
    save_file = open(output_path, mode="w")

    header = "IsBuyBoxWinner," \
             "MarketplaceId," \
             "ProductId," \
             "SellerId," \
             "ConditionNotes," \
             "IsFeaturedMerchant," \
             "IsFulfilledByAmazon," \
             "ListingPrice," \
             "ShippingPrice," \
             "ShippingTime_maxHours,ShippingTime_minHours,ShippingTime_availtype," \
             "ShipsDomestically," \
             "ShipsFromCountry,ShipsFromState," \
             "SellerFeedbackRating,SellerFeedbackCount," \
             "SubCondition," \
             "IsCustomer," \
             "Time,"\
             "CompetitionName"
    save_file.write(header + "\n")

    full_file_paths = get_filepaths(input_data_path)

    n = len(full_file_paths)
    if (n == 0):
        print("Input folder is empty! Break!")
        exit(True)
    xml_n = [f for f in full_file_paths if f.endswith('.xml')]
    if (xml_n == 0):
        print("No xml file in this folder! Break!")
        exit(True)
    fileindex = 0
    for f in (full_file_paths):
        if f.endswith(".xml"):
            fileindex = fileindex + 1

            XML_name = os.path.basename(f)
            XML_name = os.path.splitext(XML_name)[0]
            try:
                rowstr = ""
                tree = etree.parse(f)
                root = tree.getroot()
                for tags in (root.iter("NotificationMetaData")):
                    find_node = tags.find("SellerId")
                    if (find_node is not None):
                        CustomerId = str((find_node.text) + ",")
                    else:
                        CustomerId = ","

                for tags in (root.iter("OfferChangeTrigger")):
                    # print(tags)
                    find_node = tags.find("MarketplaceId")
                    if (find_node is not None):
                        MarketplaceId = str((find_node.text) + ",")
                    else:
                        MarketplaceId = ","

                    find_node = tags.find("ASIN")
                    if (find_node is not None):
                        ProductId = str((find_node.text) + ",")
                    else:
                        ProductId = ","

                    find_node = tags.find("TimeOfOfferChange")
                    if (find_node is not None):
                        TimeOffer = str((find_node.text) + ",")
                    else:
                        TimeOffer = ","
                    Time = datetime.strptime(TimeOffer, '%Y-%m-%dT%H:%M:%S.%fZ,')
                    TD = (Time - datetime(2016, 1, 1)).total_seconds()
                    TD = str(TD) + ','
                    # year = str(dt.year) + ","
                    # month = str(dt.month) + ","
                    # day = str(dt.day) + ","

                for tags in (root.iter("Offer")):
                    # print(tags)
                    find_node = tags.find("IsBuyBoxWinner")
                    if find_node.text == "true" or find_node.text == "True":
                        IsBuyBoxWinner = "1,"
                    else:
                        IsBuyBoxWinner = "-1,"

                    find_node = tags.find("SellerId")
                    if find_node is not None:
                        SellerId = str((find_node.text) + ',')
                    else:
                        SellerId = ","

                    if (SellerId == CustomerId):
                        IsCustomer = "1,"
                    else:
                        IsCustomer = "0,"

                    find_node = tags.find("ConditionNotes")
                    if (find_node is None):
                        ConditionNotes = "0,"
                    else:
                        ConditionNotes = "1,"

                    find_node = tags.find("IsFeaturedMerchant")
                    if find_node.text == "true" or find_node.text == "True":
                        IsFeaturedMerchant = "1,"
                    else:
                        IsFeaturedMerchant = "0,"

                    find_node = tags.find("IsFulfilledByAmazon")
                    if find_node.text == "true" or find_node.text == "True":
                        IsFulfilledByAmazon = "1,"
                    else:
                        IsFulfilledByAmazon = "0,"

                    find_node = tags.find("ListingPrice")
                    if (find_node is not None):
                        ListingPrice = find_node.findtext('Amount', default='None')
                        ListingPrice = ListingPrice + ","
                    else:
                        ListingPrice = "-1,"

                    find_node = tags.find("Shipping")
                    if (find_node is not None):
                        ShippingPrice = find_node.findtext('Amount', default='None')
                        ShippingPrice = ShippingPrice + ","
                    else:
                        ShippingPrice = "-1,"

                    find_node = tags.find("ShippingTime")
                    if (find_node is not None):
                        minHours = find_node.attrib["minimumHours"] + ','
                        maxHours = find_node.attrib["maximumHours"] + ','
                        availType = find_node.attrib["availabilityType"] + ','
                    else:
                        minHours = '-1,'
                        maxHours = '-1,'
                        availType = ','

                    find_node = tags.find("ShipsDomestically")
                    if find_node.text == "true" or find_node.text == "True":
                        ShipsDomestically = "1,"
                    else:
                        ShipsDomestically = "0,"

                    find_node = tags.find("ShipsFrom")
                    if (find_node is not None):
                        ShipsFromCountry = find_node.findtext('Country', default='None')
                        ShipsFromCountry = ShipsFromCountry + ","

                        ShipsFromState = find_node.findtext('State', default='None')
                        ShipsFromState = ShipsFromState + ","
                    else:
                        ShipsFromCountry = ","
                        ShipsFromState = ','

                    find_node = tags.find("SellerFeedbackRating")
                    if (find_node is not None):
                        SellerFeedbackRating = find_node.findtext('SellerPositiveFeedbackRating', default='None')
                        SellerFeedbackRating = SellerFeedbackRating + ","

                        SellerFeedbackCount = find_node.findtext('FeedbackCount', default='None')
                        SellerFeedbackCount = SellerFeedbackCount + ","
                    else:
                        SellerFeedbackRating = "-1,"
                        SellerFeedbackCount = '-1,'

                    find_node = tags.find("SubCondition")
                    if (find_node is not None):
                        SubCondition = str((find_node.text) + ',')
                    else:
                        SubCondition = ","

                    rowstr = "{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}{12}{13}{14}{15}{16}{17}{18}{19}{20}".format(
                        IsBuyBoxWinner,
                        MarketplaceId,
                        ProductId,
                        SellerId,
                        ConditionNotes,
                        IsFeaturedMerchant,
                        IsFulfilledByAmazon,
                        ListingPrice,
                        ShippingPrice,
                        maxHours, minHours,
                        availType,
                        ShipsDomestically,
                        ShipsFromCountry,
                        ShipsFromState,
                        SellerFeedbackRating,
                        SellerFeedbackCount,
                        SubCondition,
                        IsCustomer,
                        TD,
                        XML_name)
                    ## Save to file the competition
                    save_file.write(rowstr + "\n")
            except:
                error_file.write(XML_name + "\n")  # save the name of XML which fails when parsing
                pass

    error_file.close()
    save_file.close()

    df_data = pd.read_csv(output_path, error_bad_lines=False, low_memory=False)
    # df_data["Time"]  = handleTime(df_data["Time"])
    df_data.to_csv(output_path, sep=',', index=False)
    if log_line == True:
        print(
            "  --> Finished parsing " + str(len(df_data)) + " rows in to csv file at: \n  " + str(output_path))

    return df_data
## HANDLE TIME INTO VECTOR
def handleTime(timearray):
    outTime = []
    for eachtime in timearray:
        Time = datetime.strptime((eachtime), '%Y-%m-%dT%H:%M:%S.%fZ')
        TD = (Time - datetime(2016, 1, 1)).total_seconds()
        outTime.append(TD)
    return outTime

## FUNCTION TO CLEAN DATA
def cleaning(df=None, output_path=None, log_line=False):
    if (log_line == True):
        print('Cleaning ...')
    if df is None:
        if os.path.isfile('./CSV/data_raw.csv'):
            df = pd.read_csv('./CSV/data_raw.csv', error_bad_lines=False, low_memory=False)
        else:
            raise NameError('No CSV file found!')
            # ----
    if (output_path is None):
        output_path = "./"
    else:
        make_sure_path_exists(output_path)
    output_path = output_path + 'data_raw.csv'

    if (log_line == True):
        print(' Delete duplication in columns ...')
    sizeOld = df.shape[1]
    dups = duplicate_columns(df)
    df = df.drop(dups, axis=1)
    sizeNew = df.shape[1]
    no_duplicated = sizeOld - sizeNew
    # df.to_csv(output_path, sep=',', index=False)
    if (log_line == True):
        print(' --> Dimension of Raw file: ' + str(sizeOld))
        print(' --> Dimension of Clean file: ' + str(sizeNew))
        print(' --> Columns deleted: ' + str(no_duplicated))
        if (len(dups) != 0):
            print(' ' + str(dups))

    ## delete duplication rows
    if (log_line == True):
        print('\n Delete duplication in rows ...')
    sizeOld = df.shape[0]
    df.drop_duplicates(inplace=True)  # check and clean the duplication
    sizeNew = df.shape[0]
    no_duplicated = sizeOld - sizeNew

    df.to_csv(output_path, sep=',', index=False)
    if (log_line == True):
        print(' --> Size of Raw file: ' + str(sizeOld))
        print(' --> Size of Clean file: ' + str(sizeNew))
        print(' --> Rows deleted: ' + str(no_duplicated))

    ## delete constant features
    if (log_line == True):
        print('\n Delete constant columns ...')
    for column in df.columns:
        if column != 'MarketplaceId' and column != 'SellerId' and column != 'ProductId' and column != 'CompetitionName':
            if len(df[column].unique()) == 1:
                if (log_line == True):
                    print(' --> Drop [' + column + ']')
                df = df.drop(column, 1)

    ## delete >95% missing features
    if (log_line == True):
        print('\n Delete columns with many missing values (>95%) ...')
    df = df.dropna(axis=1, thresh=int(len(df) * 0.9))
    df = df.reset_index(drop=True)
    ## delete the abnormal sample (more than 2 winners, no winner)
    groups = df.groupby("CompetitionName")
    size_old = len(df)
    df_new = df.copy()
    list_del = []
    for name, group in groups:
        winner = group.loc[group.IsBuyBoxWinner == 1]
        if len(winner) != 1 : #or len(group) == 1
            list_del.append(name)
    df_new = df_new[~df_new['CompetitionName'].isin(list_del)]
    df_new = df_new.reset_index(drop=True)
    size_new = len(df_new)

    if (log_line == True):
        print('\n -----------------------------------------------')
        print('Delete rows have not one winner only')
        print(' --> Size of Raw file: ' + str(size_old))
        print(' --> Size of Clean file: ' + str(size_new))
        print(' --> Rows deleted: ' + str(size_old-size_new))
    df_new = df_new.reset_index(drop=True)
    return df_new
## FUNCTION TO GET PREPARING DATA
def preparing(df=None, log_line=False):
    if (log_line == True):
        print('Preparing Data ...')
    if df is None:
        if os.path.isfile('./CSV/data_raw.csv'):
            df = pd.read_csv('./CSV/data_raw.csv', error_bad_lines=False, low_memory=False)
        else:
            raise NameError('No CSV file found! Cannot run!')
    ##
    df_market, n_markets = splitmarket(df)

    return df_market, n_markets
## FUNCTION TO SPLIT MARKETS
def splitmarket(df=None):
    if df is None:
        if os.path.isfile('./CSV/data_raw.csv'):
            df = pd.read_csv('./CSV/data_raw.csv', error_bad_lines=False, low_memory=False)
        else:
            raise NameError('No CSV file found! Cannot run!')

    make_sure_path_exists('./CSV/Market_csv/')
    df_market = []
    group_full_market = df.groupby('MarketplaceId')

    i = 0
    TotalSize = 0
    print("  ------------------- Split markets ----------------------")
    for name, group in (group_full_market):
        TotalSize += group.shape[0]
        print("  Market" + str(i) + " [" + str(name) + "]: " + str('{:,}'.format(group.shape[0])) + " samples")
        group.to_csv("./CSV/Market_csv/" + str(name) + "_data.csv", sep=',', index=False)
        group = group.reset_index(drop=True)
        df_market.append(group)
        i = i + 1

    print("  --------------------------------------------------------")
    print('  Total:' + ":   " + str('{:,}'.format(TotalSize)) + " samples")
    return df_market, i

## FUNCTION TO LOAD MARKETS
def loadmarket(path=None):
    if path is None:
        path = './CSV/Market_csv/'

    df_market = []
    full_file_paths = get_filepaths(path)
    i = 0
    for f in (full_file_paths):
        if f.endswith(".csv"):
            i = i + 1
            df_market.append(pd.read_csv(f, error_bad_lines=False, low_memory=False))
    return df_market, i

## NEW MINNAX SCALER
def minmaxscalerb(X):
    max_ = max(X)
    min_ = min(X)
    if (max_!= min_):
        X_std = (X - min_) / (max_ - min_)
    else:
        X_std =  [1] * len(X)
    # X_scaled = X_std * (max_ - min_) + min_
    return X_std
## NORMALISE
def normalise(df, column_group, column_scale):
    # sc_func = StandardScaler()
    # sc_func = RobustScaler()
    # sc_func = MinMaxScaler()
    # print(df.groupby(column_group)[column_scale])
    df[column_scale] = df.groupby(column_group)[column_scale].transform(lambda x: minmaxscalerb(x))
    return df[column_scale]

## CLEARSCREEN (OS)
def clearscreen():
    if _platform == "linux" or _platform == "linux2":
        # linux
        os.system('clear')  # on linux / os x
    elif _platform == "darwin":
        # MAC OS X
        os.system('clear')  # on linux / os x
    elif _platform == "win32":
        # Windows
        os.system('cls')  # on windows
    elif _platform == "win64":
        # Windows
        os.system('cls')  # on windows

    print(__doc__)
### MANAGE THE CLASSIFIER MODEL
def getname(marketID):
    # currenttime = datetime.now().strftime("%Y-%m-%dz")
    filename = str(marketID)
    return filename

def classifierName(clfs):
    clfs = int(clfs)
    if (clfs == 1):
        name = 'RandomForest'  # number of tree = 10
    elif clfs == 2:
        name = 'LogisticRegression'
    elif clfs == 3:
        name = '3-NN'
    elif clfs == 4:
        name = 'SVM-Rbf'
    elif clfs == 5:
        name = 'AdaBoost'
    elif clfs ==6:
        name = 'XGBoost'
    elif clfs ==7:
        name = 'All'
    else:
        name = 'Wrong'
    return name
## SAVE
def savemodel(model, savepath, marketId,clfs):
    timeStr = time.strftime("%Y%m%d-%H%M%S")
    nameMarket = getname(marketId)
    nameClassifier = getname(clfs)
    output = open(savepath + "/" + timeStr + "_" + nameMarket + "_" + nameClassifier + ".pkl", 'wb')
    pickle.dump(model, output)
    output.close()
def savefeaturelist(featlist, savepath, marketId,clfs):
    timeStr = time.strftime("%Y%m%d-%H%M%S")
    nameMarket = getname(marketId)
    nameClassifier = getname(clfs)
    make_sure_path_exists(savepath + '/features/')
    save_file = open(savepath + "/features/" + timeStr + "_" + nameMarket + "_" + nameClassifier + ".fea", mode="w")
    save_file.write("FeatureName,FI\n")
    for item in featlist:
        save_file.write(str(item[0]) + "," + str(item[1]) + "\n")
    save_file.close()
## LOAD
def loadmodel(path):
    f = open(path, 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier
def loadfeature(path):
    df_feature = pd.read_csv(path, error_bad_lines=False, low_memory=False)
    # list_feature = df_fea.FeatureName.values
    return df_feature
## FUNCTION TO COMPUTE FEATURE IMPORTANCE
def computeFI(dataframes, n_estimators=10, bootstrap=True, plot=False, log=False):
    df = dataframes
    if 'ProductId' in df.columns:
        df = df.drop("ProductId", 1)
    if 'SellerId' in df.columns:
        df = df.drop("SellerId", 1)

    df.drop_duplicates(inplace=True)

    # to int
    if len(df.select_dtypes(['category', 'object']).columns) != 0:
        df = tonumeric(df)
    # category_columns = df.select_dtypes(['category', 'object']).columns
    # for column in category_columns:
    #     d = df[column].astype('category')
    #     d = d.cat.codes
    #     d = d.astype('float')
    #     df[column] = d
    if 'CompetitionName' in df.columns:
        df = df.drop("CompetitionName", 1)
    # df.replace(np.inf, 0)
    # df.replace(np.nan, -1)
    # print(df.info())
    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    # class_names = ['-1', '1']
    # n_samples, n_features = X.shape
    feat_labels = df.columns[1:]

    #### Train Random Forest
    # print(np.any(np.isnan(X)))
    # print(np.any(np.isfinite(X)))
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0, n_jobs=5, bootstrap=bootstrap)
    # clf = LogisticRegression()
    # clf =  SVC(kernel='linear')
    clf.fit(X, y)

    # y[y==-1]=0
    # clf = XGBClassifier()
    # clf.fit(X, y)
    ###### Compute importances
    importances = clf.feature_importances_
    # importances= clf.coef_.ravel()

    indices = np.argsort(importances)[::-1]
    fealist = []
    for f in range(X.shape[1]):
        if (log == True):
            print("%2d) %-*s \t %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
        fealist.append([feat_labels[indices[f]], importances[indices[f]]])

    # # plot
    # if (plot == True):
    #     plt.figure(figsize=(10, 8))
    #     plt.title('Features Importances')
    #     plt.bar(range(X.shape[1]), importances[indices], color='lightblue', align='center')
    #     plt.xticks(range(X.shape[1]), feat_labels[indices], rotation=90)
    #     plt.xlim([-1, X.shape[1]])
    #     plt.tight_layout()
    #     plt.show(block=False)

    return fealist
## TO ASCII
def convert_to_ascii(text):
    return "".join(str(ord(char)) for char in text)
## TO NUMERIC
def tonumeric(df):
    category_columns = df.select_dtypes(['category', 'object']).columns
    # Change the unique string into integer
    for column in category_columns:
        converted_value = []
        d = df[column].astype('category')
        for each in d:
            if(each == ""):
                converted_value.append(-1)
            else:
                converted_value.append(np.log(float(convert_to_ascii(each))))
        df[column] = pd.Series((float(v) for v in converted_value))
    return df
## ADD LANDED PRICE FEATURES
def addPricefeatures(dataframe, norm=False):
    df = dataframe
    ## scale data into same scaled space
    sc_func = MinMaxScaler()
    # sc_func = StandardScaler()
    #     sc_func = RobustScaler()

    if ('LandedPrice' in df.columns):
        pass
    else:
        df['LandedPrice'] = df['ListingPrice'] + df['ShippingPrice']
    df['DifftoMinLandedPriceCompetition'] = df.groupby('CompetitionName')[['LandedPrice']].transform(
        lambda x: (x - min(x)))
    if (norm == True):
        df['ListingPrice'] = df.groupby('CompetitionName')[['ListingPrice']].transform(
            lambda x: sc_func.fit_transform(x))
        df['ShippingPrice'] = dataframe.groupby('CompetitionName')[['ShippingPrice']].transform(
            lambda x: sc_func.fit_transform(x))
        df['LandedPrice'] = df.groupby('CompetitionName')[['LandedPrice']].transform(lambda x: sc_func.fit_transform(x))
        df['DifftoMinLandedPriceCompetition'] = df.groupby('CompetitionName')[[
            'DifftoMinLandedPriceCompetition']].transform(lambda x: sc_func.fit_transform(x))
    return df
## SCORE F1 RESULTS
def scoring(predicted, y_test, log=False):
    precision, recall, fscore, support = score(y_test, predicted)
    if log == True:
        print('\nClassification Report:\n', metrics.classification_report(y_test, predicted))
        print('-------------------------------------------------------------------------------')
    return precision, recall, fscore, support
## SPLIT DATA INTO TRAIN-TEST
def splitdata(df, test_ratio=0.3,truedata=None):
    n = len(df.CompetitionName.unique())
    n_test = round(test_ratio * n)
    df_test = df.loc[df['CompetitionName'].isin(df.CompetitionName.unique()[:n_test])]  ## good way!
    df_train = df.loc[df['CompetitionName'].isin(df.CompetitionName.unique()[n_test:])]
    if not (truedata is None):
        true_test = truedata.loc[df['CompetitionName'].isin(df.CompetitionName.unique()[:n_test])]  ## good way!
        true_train = truedata.loc[df['CompetitionName'].isin(df.CompetitionName.unique()[n_test:])]

        # print([len(df_test),len(true_test)])
        return df_train, df_test, true_train, true_test
    else:
        return df_train, df_test
## TRY TRAIN AND TEST
def TrainTestResults2(trainset, testset, i, fealist, model=None, log=False,true_train=None, true_test=None):
    ### Top 5 best feature cut for original data
    tmp_list = np.array(fealist)[:, 0]
    top_fea = np.append(['IsBuyBoxWinner'], tmp_list[:i])

    if len(trainset.select_dtypes(['category', 'object']).columns) != 0:
        trainset = tonumeric(trainset)
        testset = tonumeric(testset)

    trainset_cut = trainset.loc[:, (top_fea)]
    testset_cut = testset.loc[:, (top_fea)]

    if not (true_train is None):
        true_train_cut = true_train.loc[:, (top_fea)]
        true_test_cut = true_test.loc[:, (top_fea)]

    x_train1, y_train1 = trainset_cut.iloc[:, 1:].values, trainset_cut.iloc[:, 0].values
    x_test1, y_test1 = testset_cut.iloc[:, 1:].values, testset_cut.iloc[:, 0].values

    if model == None:
        rfc1 = RandomForestClassifier(n_estimators=10)
        rfc1.fit(x_train1, y_train1)
        # y_train1[y_train1 == -1] = 0
        # rfc1 = XGBClassifier()
        # rfc1.fit(x_train1, y_train1)
    else:
        rfc1 = model

    # Model 1 - test 1
    # y_predicted1 = cross_val_predict(rfc1, x_test1, y_test1, cv=10)
    tmp_pro = rfc1.predict(x_test1)
    y_predicted1 = [round(value) for value in tmp_pro]

    # y_predicted1 = [1 if tmp_pro > 0.5 else 0]

    # true_test["prob"] = list(tmp_pro[:, 1])
    true_test["prob"] = list(tmp_pro)
    true_test = decisionmaking(true_test)
    winner = (true_test["NewDecision"].astype("int"))
    # y_predicted1, tmp_pro = applying(rfc1, x_test1)
    # class_prob = tmp_pro[:, 1] * 100
    class_prob = tmp_pro * 100
    # if not (true_train is None):
    #     # print([len(true_test_cut),len(y_predicted1)])
    #     true_test_cut["Predicted"] = list(y_predicted1)
    #     true_test_cut["Probabilities"] = list(class_prob)
    #     true_test_cut["SellerId"] = true_test["SellerId"]
    #     true_test_cut["CompetitionName"] = true_test["CompetitionName"]
    #     true_test_cut["NewScore"] = list(true_test["NewScore"])
    #     true_test_cut["NewDecision"] = list(true_test["NewDecision"])
    #     true_test_cut["max"] = list(true_test["max"])

    # precision, recall, fscore, support = scoring(y_predicted1, y_test1, log)
    precision2, recall2, fscore2, support2 = scoring(winner, y_test1, log)
    # cnf_matrix1 = confusion_matrix(y_test1, y_predicted1)
    cnf_matrix2 = confusion_matrix(y_test1, winner)
    if (log == True):
        # print(cnf_matrix1)
        print(cnf_matrix2)
        # if not (true_train is None):
            # printresults(data=true_test_cut)
    return fscore2, rfc1
## PRINT THE RESULTS OF TESTDATA
def printresults(data):
    clearscreen()
    # print("\n{0} \t\t {1} \t\t {2} \t\t {3} \t\t {4}".format("CustomerID", "Price", "TrueLabel", "Prediction", "Probability"))
    print("--------------------------------------------------------------------------")
    groups = data.groupby("CompetitionName")
    for name, group in groups:
        tmp = group.drop('CompetitionName', 1)
        # cols = list(tmp)
        # cols.insert(0, cols.pop(cols.index('SellerId')))
        # cols.insert(-1, cols.pop(cols.index('IsBuyBoxWinner')))
        # cols.insert(-1, cols.pop(cols.index('Predicted')))
        # cols.insert(-1, cols.pop(cols.index('Probabilities')))
        # tmp = tmp.reindex(columns=cols)
        # tmp.rename(columns={col: "" for col in tmp})
        pd.options.display.width = 50000
        pd.options.display.max_colwidth = 50000
        # print(tmp)
        print("--------------------------------------------------------------------------")
## ADD THE FEATURES OF AMAZON SELLER TO DATAFRAME
def addAMZfeatures(dataframe, amzId, norm=False):
    newdf = dataframe
    newdf['DifftoAmzLandedPriceCompetition'] = [-9999] * len(newdf)
    newdf['CompeteAmzCompetition'] = newdf.groupby('CompetitionName')[['SellerId']].transform(
        lambda x: 1 if (sum((x == amzId) * 1) != 0) else -1)
    groups = newdf[newdf['CompeteAmzCompetition'] == 1].groupby('CompetitionName')
    #     groups = newdf.groupby(['CompetitionName'])
    n = len(groups)
    i = 1
    for name, group in groups:
        # print('Working in group (AMZ): ' + str(i) + '/' + str(n), end='\r')
        gidx = group.index
        amzp = min(group[group.SellerId == amzId].LandedPrice)
        newdf['DifftoAmzLandedPriceCompetition'].iloc[gidx] = group['LandedPrice'] - amzp
        i = i + 1

    if (norm == True):
        newdf['ListingPrice'] = normalise(newdf, 'CompetitionName', 'ListingPrice')
        newdf['ShippingPrice'] = normalise(newdf, 'CompetitionName', 'ShippingPrice')
        newdf['LandedPrice'] = normalise(newdf, 'CompetitionName', 'LandedPrice')

    newdf.loc[newdf['SellerId'] == amzId, 'CompeteAmzCompetition'] = 0
    return newdf
## ADD THE FEATURES  DATAFRAME
def addMinProdfeatures(dataframe, listmin={}, norm=False):
    newdf = dataframe
    newlistmin = {}
    if ('DifftoMinLandedPriceProduct' in dataframe.columns):
        pass
    else:
        newdf['DifftoMinLandedPriceProduct'] = [-9999] * len(newdf)
        groups = newdf.groupby('ProductId')
        n = len(groups)
        for name, group in groups:
            # print('Working in group (MinProd): ' + str(i) + '/' + str(n), end='\r')
            min_tmp = float(min(group['LandedPrice']))
            gidx = group.index
            if(not listmin):
                minp = min(group.LandedPrice)
                newdf['DifftoMinLandedPriceProduct'].iloc[gidx] = group['LandedPrice'] - minp
                newlistmin.update({str(name): minp})
            else:
                tmpSeries = group.ProductId
                minp = float(tmpSeries.map(listmin).unique())
                if (np.isnan(minp)):
                    minp = min_tmp
                newdf['DifftoMinLandedPriceProduct'].iloc[gidx] = group['LandedPrice'] - minp
        if (not listmin):
            pass
        else:
            newlistmin = listmin

    # newdf['DifftoMinLandedPriceProduct'] = normalise(newdf,'ProductId','DifftoMinLandedPriceProduct')
    if (norm == True):
        newdf['ListingPrice'] = normalise(newdf, 'CompetitionName', 'ListingPrice')
        newdf['ShippingPrice'] = normalise(newdf, 'CompetitionName', 'ShippingPrice')
        newdf['LandedPrice'] = normalise(newdf, 'CompetitionName', 'LandedPrice')

    return newdf, newlistmin
## DISSIMILARITY MATRIX COMPUTATION
def gower_distances(X, Y=None, w=None):
    """
    Computes the gower distances between X and Y

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    Y : array-like, shape (n_samples, n_features)

    Returns
    -------
    distances : ndarray, shape (n_samples, )

    Notes
    ------
    Gower is a similarity for categorical, boolean and numerical mixed data.
    """

    X, Y = pairwise.check_pairwise_arrays(X, Y, dtype=np.object)

    rows, cols = X.shape
    dtypes = []
    for col in range(cols):
        dtypes.append(type(X[0, col]))

    # calculate the range and max values of numeric values for mixed data
    ranges_of_numeric = [0.0] * cols
    max_of_numeric = [0.0] * cols
    for col in range(cols):
        if np.issubdtype(dtypes[col], np.number):
            max = np.nanmax(X[:, col].astype(dtypes[col])) + 0.0
            if np.isnan(max):
                max = 0.0

            max_of_numeric[col] = max

            min = np.nanmin(X[:, col].astype(dtypes[col])) + 0.0
            if np.isnan(min):
                min = 0.0

            ranges_of_numeric[col] = (1 - min / max, 0)[max == 0]

    # According the Gower formula, w is an attribute weight
    if w is None:
        w = [1] * cols

    yrows, ycols = Y.shape

    dm = np.zeros((rows, yrows), dtype=np.double)

    for i in range(0, rows):
        j_start = i

        # for non square results
        if rows != yrows:
            j_start = 0

        for j in range(j_start, yrows):
            xi = X[i]
            xj = Y[j]
            sum_sij = 0.0
            sum_wij = 0.0
            for col in range(cols):
                value_xi = xi[col]
                value_xj = xj[col]
                if np.issubdtype(dtypes[col], np.number):
                    if (max_of_numeric[col] != 0):
                        value_xi = value_xi / max_of_numeric[col]
                        value_xj = value_xj / max_of_numeric[col]
                    else:
                        value_xi = 0
                        value_xj = 0

                    if ranges_of_numeric[col] != 0:
                        sij = abs(value_xi - value_xj) / ranges_of_numeric[col]
                    else:
                        sij = 0
                    wij = (w[col], 0)[np.isnan(value_xi) or np.isnan(value_xj)]
                else:
                    sij = (1.0, 0.0)[value_xi == value_xj]
                    wij = (w[col], 0)[value_xi is None and value_xj is None]
                sum_sij += (wij * sij)
                sum_wij += wij

            if sum_wij != 0:
                dm[i, j] = (sum_sij / sum_wij)
                if j < rows and i < yrows:
                    dm[j, i] = dm[i, j]

    return dm
## ADD THE FEATURES  DATAFRAME
def addCombinationfeatures(dataframe, norm=False):
    newdf = dataframe.copy()
    if ('IdealPointCompetition' in dataframe.columns):
        pass
    else:
        #     newdf['IdealPointCompetition'] = newdf['DifftoMinLandedPriceCompetition']
        newdf['IdealPointCompetition'] = [-9999] * len(newdf)
        groups = newdf.groupby(['CompetitionName'])
        n = len(groups)
        i = 1
        for name, group in groups:
            # print('Working in group (Comb): ' + str(i) + '/' + str(n), end='\r')
            gidx = group.index
            ## get the top ideal profile
            # tmp_group = group[['LandedPrice','ShippingTime_maxHours','SellerFeedbackCount','SellerFeedbackRating','IsFulfilledByAmazon','IsFeaturedMerchant','CompeteAmzCompetition']].copy()

            # tmp_group = group[['LandedPrice','ShippingTime_maxHours']].copy()
            # df_idealProfile = tmp_group.head(1)

            # df_idealProfile.LandedPrice = min(tmp_group.LandedPrice)
            # df_idealProfile.ShippingTime_maxHours = min(tmp_group.ShippingTime_maxHours)
            # df_idealProfile.SellerFeedbackCount = max(tmp_group.SellerFeedbackCount)
            # df_idealProfile.SellerFeedbackRating = max(tmp_group.SellerFeedbackRating)
            # df_idealProfile.IsFulfilledByAmazon = 1
            # df_idealProfile.IsFeaturedMerchant = 1
            # df_idealProfile.CompeteAmzCompetition = 0

            ## Add the group profiles
            # df_idealProfile = df_idealProfile.append(tmp_group,ignore_index=True)

            ## Calculate the dissimilarity
            # D = gower_distances(df_idealProfile)
            # vector_dissimilarity =  D[1:-1,0]
            # diff_value = vector_dissimilarity

            # OLD COMPUTATION
            vector_LandedPrice = (group.LandedPrice)
            vector_ShippingTime_maxHours = (group.ShippingTime_maxHours)
            vector_Feedback = (group.SellerFeedbackCount * group.SellerFeedbackRating)
            # print([vector_LandedPrice,vector_ShippingTime_maxHours,vector_Feedback])
            min_Lprice = min(vector_LandedPrice)
            min_Maxtime = min(vector_ShippingTime_maxHours)
            max_feedback = max(vector_Feedback)
            # print([min_Lprice, min_Maxtime, max_feedback])
            diff_value = (min_Lprice*min_Maxtime -vector_LandedPrice*vector_ShippingTime_maxHours)

            newdf['IdealPointCompetition'].iloc[gidx] = diff_value
            # print(newdf)
            i = i + 1
    # newdf['IdealPointCompetition'] = normalise(newdf,'CompetitionName','IdealPointCompetition')
    if (norm == True):
        newdf['ListingPrice'] = normalise(newdf, 'CompetitionName', 'ListingPrice')
        newdf['ShippingPrice'] = normalise(newdf, 'CompetitionName', 'ShippingPrice')
        newdf['LandedPrice'] = normalise(newdf, 'CompetitionName', 'LandedPrice')
    return newdf
## COMPUTE THE THRESHOLD FOR NUMBER OF FEATURE
def thresholdcomputing(trainset, testset, fealist_nt):
    ## Draw the thresholdD:\git\ML4Amazon\code\PackageCode\ForUs\StandardCode
    fscore_opti = 0
    best_N_nt = 1
    bestmodel_nt = None
    fscore_nt = []
    for i in np.arange(len(fealist_nt)):
        fscore, model = TrainTestResults2(trainset, testset, i + 1, fealist_nt, None, False,None,None)
        fscore_nt.append(fscore)
        if (fscore_opti < np.prod(fscore)):
            fscore_opti = np.prod(fscore)
            best_N_nt = i + 1
            bestmodel_nt = model
    print('Best cut at: ' + str(best_N_nt))
    fscore_nt = np.reshape(fscore_nt, (len(fealist_nt), 2))
    return best_N_nt, fscore_nt, bestmodel_nt
## SPLIT RANGE FROM MIN TO MAX WITH STEPS
def splitrange(min, max, current_value, steps, current_state):
    midle_points = int(steps / 2)
    last_point = steps - midle_points
    if (current_state == -1):
        if (min < current_value):
            range = np.linspace(min, current_value, steps, endpoint=True)
        else:
            range = np.linspace(current_value, min, steps, endpoint=True)
    if (current_state == 1):
        if (max > current_value):
            range = np.linspace(current_value, max, steps, endpoint=True)
        else:
            range = np.linspace(max, current_value, steps, endpoint=True)
    # return np.concatenate((range1,range2), axis=0)
    return range
## GET THE AMZ SELLER ID FOR MARKET
def getAMZ(marketId):
    if marketId == 'A13V1IB3VIYZZH':
        return 'A1X6FK5RDHNB96'  # FR
    elif marketId == 'A1F83G8C2ARO7P':
        return 'A3P5ROKL5A1OLE'  # UK
    elif marketId == 'A1PA6795UKMFR9':
        return 'A3JWKAKR8XB7XF'  # DE
    elif marketId == 'A1RKKUPIHCS9HS':
        return 'A1AT7YVPFBWXBL'  # ES
    elif marketId == 'A21TJRUUN4KGV':
        return 'A380B3WNC3LCZM'  # IN
    elif marketId == 'APJ6JRA9NG5V4':
        return 'A11IL2PNWYJU7H'  # IT
    elif marketId == 'ATVPDKIKX0DER':
        return 'ATVPDKIKX0DER'  # US
    elif marketId == 'A1VC38T7YXB528':
        return 'AN1VRQENFRJN5'  # JP
    elif marketId == 'A2EUQ1WTGCTBG2':
        return 'A3DWYIK6Y9EEQB'  # CA
    else:
        return 'nothing'
##### CHECK FEATURE IMPORTANCE
def checkFI(df, amzId, plotfi=False, confusion_results=True, log_line=False, test_ratio=0.3, truedata=None):
    if (log_line == True):
        print('Delete features ...')
    if 'ListingCurrency' in df.columns:
        df = df.drop('ListingCurrency', 1)
        truedata = truedata.drop('ListingCurrency', 1)
    if 'ShippingCurrency' in df.columns:
        df = df.drop('ShippingCurrency', 1)
        truedata = truedata.drop('ShippingCurrency', 1)
    if 'MarketplaceId' in df.columns:
        df = df.drop('MarketplaceId', 1)
        truedata = truedata.drop('MarketplaceId', 1)
    if 'ShipsDomestically' in df.columns:
        df = df.drop('ShipsDomestically', 1)
        truedata = truedata.drop('ShipsDomestically', 1)

    # df_ = addPricefeatures(df, norm=False)
    # df_ = addAMZfeatures(df_, amzId, norm=False)
    # df_.loc[df_.ShippingTime_maxHours == 0, 'ShippingTime_maxHours'] = 1
    # df_ = addCombinationfeatures(df_, norm=False)

    df_ = df.reset_index(drop=True)
    truedata = truedata.reset_index(drop=True)

    trainset_, testset_, true_train , true_test = splitdata(df_, test_ratio, truedata)
    trainset_ = trainset_.reset_index(drop=True)
    testset_ = testset_.reset_index(drop=True)

    ## truetrain and truetest is the raw data
    true_train = true_train.reset_index(drop=True)
    true_test = true_test.reset_index(drop=True)

    ## add minproduct
    # trainset_,listmin, = addMinProdfeatures(trainset_,[], norm=True)
    # testset_ = addMinProdfeatures(testset_,listmin, norm=True)

    ## normalize
    # trainset_['IdealPointCompetition'] = normalise(trainset_, 'CompetitionName', 'IdealPointCompetition')
    # testset_['IdealPointCompetition'] = normalise(testset_, 'CompetitionName', 'IdealPointCompetition')
    # trainset_['DifftoMinLandedPriceProduct'] = normalise(trainset_, 'ProductId', 'DifftoMinLandedPriceProduct')
    # testset_['DifftoMinLandedPriceProduct'] = normalise(testset_, 'ProductId', 'DifftoMinLandedPriceProduct')

    if (log_line == True):
        print('==================================================')
        print('Split train,test')
        print('Size of train set:', len(trainset_))
        print('Size of testing set:', len(testset_))
        print('==================================================')

    ## delete correlation columns
    if 'ListingPrice' in trainset_.columns:
        trainset_ = trainset_.drop('ListingPrice', 1)
        testset_ = testset_.drop('ListingPrice', 1)
        true_train = true_train.drop('ListingPrice', 1)
        true_test = true_test.drop('ListingPrice', 1)
    if 'ShippingPrice' in trainset_.columns:
        trainset_ = trainset_.drop('ShippingPrice', 1)
        testset_ = testset_.drop('ShippingPrice', 1)
        true_train = true_train.drop('ShippingPrice', 1)
        true_test = true_test.drop('ShippingPrice', 1)
    if 'ShippingTime_minHours' in trainset_.columns:
        trainset_ = trainset_.drop('ShippingTime_minHours', 1)
        testset_ = testset_.drop('ShippingTime_minHours', 1)
        true_train = true_train.drop('ShippingTime_minHours', 1)
        true_test = true_test.drop('ShippingTime_minHours', 1)

    ## concat train+test
    # if (log_line == True):
    #     print('==================================================')

    ## threshold to select features
    fealist_comb3 = computeFI(trainset_, 10, True, plot=plotfi)
    if (log_line == True):
        print('Feature Importance: Done!')
        print('==================================================')

    ## threshold to select features
    # best_N_comb3, fscore_comb3, bestmodel_comb3 = \
    #     thresholdcomputing(trainset_, testset_, fealist_comb3)
    # thresholdplotting(fealist_comb3, fscore_comb3, best_N_comb3)
    best_N_comb3 = len(df.columns) ## keep always 10 features in feature list

    # if (log_line == True):
    #     print('Threshold plotting: Done!')
    #     print('==================================================')

    if (confusion_results == True):
        f1_comb3, model_comb3 = \
            TrainTestResults2(trainset_, testset_, best_N_comb3, fealist_comb3, model=None, log=True, true_train=true_train, true_test=true_test)
        print('Result score: Done!')
        print('==================================================')
        return fealist_comb3, best_N_comb3, f1_comb3, model_comb3
    else:
        return fealist_comb3, best_N_comb3
## INPUT CHECKING
def restricted_float(x):
    x = float(x)
    if x < 0.0 or x >= 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0)" % (x,))
    return x
def restricted_int(x):
    x = int(x)
    if x < 1 or x >= 8:
        raise argparse.ArgumentTypeError("%r not in range [1, 7]" % (x,))
    return x
def restricted_bool(x):
    if x not in ['True', 'False', 'T', 'F', 'true', 'false', 'TRUE', 'FALSE', 't', 'f']:
        raise argparse.ArgumentTypeError("%r is not True|False " % (x,))
    return x
## DEFINE THE CLASSIFIER
def classifierchoose(clfs):
    clfs = int(clfs)
    if (clfs ==1):
        model_ = RandomForestClassifier(n_estimators=10)  # number of tree = 10
    elif clfs ==2:
        model_ = LogisticRegression()
    elif clfs ==3:
        model_ = KNeighborsClassifier(3)
    elif clfs == 4:
        model_ = SVC(kernel="rbf", probability=True)
    elif clfs == 5:
        model_ = AdaBoostClassifier()
    elif clfs == 6:
        model_ = XGBClassifier()
    return model_
## TRAIN - TEST AND APPLY
def training(traindata, trainclass,clfs=1):
    # trainclass[trainclass == -1] = 0
    # model_ = XGBClassifier()
    # model_.fit(traindata, trainclass)
    model_ = classifierchoose(clfs)
    model_.fit(traindata, trainclass)
    return model_
def testting(model_, testdata, testclass):
    tmp_pro = model_.predict(testdata)
    predictedclass = [round(value) for value in tmp_pro]
    # predictedclass = model_.predict(testdata)
    precision, recall, fscore, support = scoring(predictedclass, testclass, log=False)
    cnf_matrix = confusion_matrix(testclass, predictedclass)
    return precision, recall, fscore, support, cnf_matrix
def applying(model_, applydata):
    predictedclass = model_.predict(applydata)
    class_prob = model_.predict_proba(applydata)
    return predictedclass, class_prob
## GUI FUNCTIONS
# def openfolder(initialdir="./", title='Please select'):
#     root = Tkinter.Tk()
#     root.title('Directory: XMLs data')
#     root.withdraw()
#     root.update()
#     path = tkFileDialog.askdirectory(parent=root, initialdir=initialdir,
#                                      title=title)
#     return path

# def openfile(initialdir="./", title='Please select the file', filetypes=[('All files', '*')]):
#     root = Tkinter.Tk()
#     root.withdraw()
#     root.update()
#     file_path = tkFileDialog.askopenfilename(initialdir=initialdir, title=title, filetypes=filetypes)
#     return file_path

## MANAGE THE DATA (ADD FEATURES ...)
def datamanage(df, test_ratio, fealist_, best_N_, amzId,listmin={}):
    if 'ListingCurrency' in df.columns:
        df = df.drop('ListingCurrency', 1)
    if 'ShippingCurrency' in df.columns:
        df = df.drop('ShippingCurrency', 1)
    if 'MarketplaceId' in df.columns:
        df = df.drop('MarketplaceId', 1)
    if 'ShipsDomestically' in df.columns:
        df = df.drop('ShipsDomestically', 1)

    tmp_list = np.array(fealist_)[:, 0]
    top_feature = np.append(['IsBuyBoxWinner'], tmp_list[:best_N_])
    # top_feature = np.append(['IsFeaturedMerchant'], tmp_list[:best_N_])
    df = df.reset_index(drop=True)

    # print('Delete features: Done!')
    df_ = addPricefeatures(df, norm=False)
    df_ = addAMZfeatures(df_, amzId, norm=False)
    df_.loc[df_.ShippingTime_maxHours == 0, 'ShippingTime_maxHours'] = 1
    df_ = addCombinationfeatures(df_, norm=False)

    df_ = df_.reset_index(drop=True)

    if (test_ratio != 0):
        trainset_, testset_ = splitdata(df_, test_ratio,None)
        trainset_ = trainset_.reset_index(drop=True)
        testset_ = testset_.reset_index(drop=True)

        ## add minproduct
        trainset_,listmin_ = addMinProdfeatures(trainset_,listmin={}, norm=True)
        testset_,listmin_ = addMinProdfeatures(testset_,listmin=listmin_, norm=True)

        ## normalize
        trainset_['IdealPointCompetition'] = normalise(trainset_, 'CompetitionName', 'IdealPointCompetition')
        testset_['IdealPointCompetition'] = normalise(testset_, 'CompetitionName', 'IdealPointCompetition')
        trainset_['DifftoMinLandedPriceProduct'] = normalise(trainset_, 'ProductId', 'DifftoMinLandedPriceProduct')
        testset_['DifftoMinLandedPriceProduct'] = normalise(testset_, 'ProductId', 'DifftoMinLandedPriceProduct')

        ## delete correlation columns
        if 'ListingPrice' in trainset_.columns:
            trainset_ = trainset_.drop('ListingPrice', 1)
            testset_ = testset_.drop('ListingPrice', 1)
        if 'ShippingPrice' in trainset_.columns:
            trainset_ = trainset_.drop('ShippingPrice', 1)
            testset_ = testset_.drop('ShippingPrice', 1)
        if 'ShippingTime_minHours' in trainset_.columns:
            trainset_ = trainset_.drop('ShippingTime_minHours', 1)
            testset_ = testset_.drop('ShippingTime_minHours', 1)

        trainset__tmp = trainset_.loc[:, top_feature]
        if len(trainset__tmp.select_dtypes(['category', 'object']).columns) != 0:
            trainset_ = tonumeric(trainset__tmp)
        testset__tmp = testset_.loc[:, top_feature]
        if len(testset__tmp.select_dtypes(['category', 'object']).columns) != 0:
            testset_ = tonumeric(testset__tmp)
        del trainset__tmp, testset__tmp

        return trainset_, testset_
    else:
        ## add minproduct
        trainset_ = df_.copy()
        trainset_ = trainset_.reset_index(drop=True)
        trainset_,listmin_ = addMinProdfeatures(trainset_,listmin=listmin, norm=True)

        ## normalize
        trainset_['IdealPointCompetition'] = normalise(trainset_, 'CompetitionName', 'IdealPointCompetition')
        trainset_['DifftoMinLandedPriceProduct'] = normalise(trainset_, 'ProductId', 'DifftoMinLandedPriceProduct')

        ## delete correlation columns
        if 'ListingPrice' in trainset_.columns:
            trainset_ = trainset_.drop('ListingPrice', 1)
        if 'ShippingPrice' in trainset_.columns:
            trainset_ = trainset_.drop('ShippingPrice', 1)
        if 'ShippingTime_minHours' in trainset_.columns:
            trainset_ = trainset_.drop('ShippingTime_minHours', 1)

        trainset_ = trainset_.loc[:, top_feature]
        if len(trainset_.select_dtypes(['category', 'object']).columns) != 0:
            trainset_ = tonumeric(trainset_)

        return trainset_,listmin_
        #######
## ADD FEATURES FUNCTION
def dataadd(df, amzId, test_ratio=0,listmin={}):

    if 'ListingCurrency' in df.columns:
        df = df.drop('ListingCurrency', 1)
    if 'ShippingCurrency' in df.columns:
        df = df.drop('ShippingCurrency', 1)
    if 'MarketplaceId' in df.columns:
        df = df.drop('MarketplaceId', 1)
    if 'ShipsDomestically' in df.columns:
        df = df.drop('ShipsDomestically', 1)

    # print('Delete features: Done!')
    df_ = addPricefeatures(df, norm=False)
    df_ = addAMZfeatures(df_, amzId, norm=False)
    df_.loc[df_.ShippingTime_maxHours == 0, 'ShippingTime_maxHours'] = 1
    df_ = addCombinationfeatures(df_, norm=False)

    df_ = df_.reset_index(drop=True)

    if (test_ratio != 0):
        trainset_, testset_= splitdata(df_, test_ratio,None)
        trainset_ = trainset_.reset_index(drop=True)
        testset_ = testset_.reset_index(drop=True)

        ## add minproduct
        trainset_,listmin_ = addMinProdfeatures(trainset_,listmin=listmin, norm=True)
        testset_,listmin_ = addMinProdfeatures(testset_,listmin=listmin_, norm=True)

        ## normalize
        trainset_['IdealPointCompetition'] = normalise(trainset_, 'CompetitionName', 'IdealPointCompetition')
        testset_['IdealPointCompetition'] = normalise(testset_, 'CompetitionName', 'IdealPointCompetition')
        trainset_['DifftoMinLandedPriceProduct'] = normalise(trainset_, 'ProductId', 'DifftoMinLandedPriceProduct')
        testset_['DifftoMinLandedPriceProduct'] = normalise(testset_, 'ProductId', 'DifftoMinLandedPriceProduct')

        ## delete correlation columns
        if 'ListingPrice' in trainset_.columns:
            trainset_ = trainset_.drop('ListingPrice', 1)
            testset_ = testset_.drop('ListingPrice', 1)
        if 'ShippingPrice' in trainset_.columns:
            trainset_ = trainset_.drop('ShippingPrice', 1)
            testset_ = testset_.drop('ShippingPrice', 1)
        if 'ShippingTime_minHours' in trainset_.columns:
            trainset_ = trainset_.drop('ShippingTime_minHours', 1)
            testset_ = testset_.drop('ShippingTime_minHours', 1)
        if len(trainset_.select_dtypes(['category', 'object']).columns) != 0:
            trainset_ = tonumeric(trainset_)
            testset_ = tonumeric(testset_)

        for cl in trainset_.columns:
            if cl != 'IsBuyBoxWinner' and cl != 'DifftoMinLandedPriceProduct' and cl != 'CompetitionName' and cl != 'ProductId':
                trainset_[cl] = normalise(trainset_,'CompetitionName',cl)
                testset_[cl] = normalise(testset_, 'CompetitionName', cl)
        return trainset_, testset_, listmin_
    else:
        ## add minproduct
        trainset_ = df_.copy()
        trainset_ = trainset_.reset_index(drop=True)
        trainset_, listmin_ = addMinProdfeatures(trainset_,listmin=listmin, norm=True)

        ## normalize
        trainset_['IdealPointCompetition'] = normalise(trainset_, 'CompetitionName', 'IdealPointCompetition')
        trainset_['DifftoMinLandedPriceProduct'] = normalise(trainset_, 'ProductId', 'DifftoMinLandedPriceProduct')

        ## delete correlation columns
        if 'ListingPrice' in trainset_.columns:
            trainset_ = trainset_.drop('ListingPrice', 1)
        if 'ShippingPrice' in trainset_.columns:
            trainset_ = trainset_.drop('ShippingPrice', 1)
        if 'ShippingTime_minHours' in trainset_.columns:
            trainset_ = trainset_.drop('ShippingTime_minHours', 1)
        if len(trainset_.select_dtypes(['category', 'object']).columns) != 0:
            trainset_ = tonumeric(trainset_)

        for cl in trainset_.columns:
            if cl != 'IsBuyBoxWinner' and cl != 'DifftoMinLandedPriceProduct' and cl != 'CompetitionName' and cl != 'ProductId':
                trainset_[cl] = normalise(trainset_,'CompetitionName',cl)
        return trainset_,listmin_
        #######
## CUT DATA BY THRESHOLD
def datacut(df, fealist_, best_N_):
    tmp_list = np.array(fealist_)[:, 0]
    top_feature = np.append(['IsBuyBoxWinner'], tmp_list[:best_N_])
    df_ = df.reset_index(drop=True)

    df_ = df_.loc[:, top_feature]

    return df_
## PRESS KEY TO CONTINUE
def presskey(Noticestring):
    try:
        input(Noticestring)
    except SyntaxError:
        pass
## NEAREST TO SOME PIVOT POINT
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))
## EXTRACT NAME OF MARKET TO GET MODEL
def extractname(mkID, file_path):
    out_name = []
    for name in glob.glob(file_path + "/*_" + str(mkID[0]) + '*.*'):
        out_name.append(name)

    # print("\t The feature list and model: \n \t" + str(out_name))
    return out_name
## GET THE EXTENSION OF FILE
def getextension(path):
    extension = os.path.splitext(path)[1]
    return extension
## MAIN FUNCTION TO DISPLAY RESULTS
def result_display(customer_id, customer_price, customer_status, list_price, class_prob, df_notcustomer, winpoint,plotforus=False,flag=False,nameXML="fig",winner_score=1,customer_score=0,name_clf='RandomForest', log_line=False):
    # clearscreen()
    # if (plotforus==True):di
    if log_line == True:
        timeStr = time.strftime("%Y%m%d-%H%M%S")
        output_path = "./OutputLog/"
        make_sure_path_exists(output_path)
        saved = sys.stdout
        if os.path.exists(str(output_path) + "/" + str(nameXML) + '.log'):
            fout = open(str(output_path) + "/" + str(nameXML) + '.log', 'a')  # File where you need to keep the logs . append
        else:
            fout = open(str(output_path) + "/" + str(nameXML) + '.log', 'w')  # File where you need to keep the logs . new
        sys.stdout = writer(sys.stdout, fout)
    if log_line == True:
        print("\n========================== Details ===============================")
        print('\n Using classifier: ' + str(name_clf) + '| Time of execution=' + str(timeStr) + ' \n')
        print("\n------------------------- Profile --------------------------------")

    if (customer_status == 1):
        customer_status = "Win"
    else:
        customer_status = "Lose"
    if log_line == True:
        print("{0:20}  {1:10}  {2:8}  {3:20}".format("CustomerID", "Price", "Status" , "PredictedScore(%)"))
        print("------------------------------------------------------------------")
        print("{0:20}  {1:10}  {2:8}  {3:20}".format(str(customer_id), '{0:.2f}'.format(np.around(customer_price, decimals=3)),
                                                      str(customer_status),'{0:.2f}'.format(np.around(customer_score, decimals=3))))
        print("------------------------------------------------------------------")

    for row in df_notcustomer.itertuples():
        if (row.IsBuyBoxWinner == 1):
            each_status = "Win"
        else:
            each_status = "Lose"
        if log_line == True:
            print("{0:20}  {1:10}  {2:8}  {3:20}".format(str("Competitor"),
                                                      '{0:.2f}'.format(np.around(row.ListingPrice + row.ShippingPrice, decimals=3)),
                                                      str(each_status),'{0:.2f}'.format(np.around(row.NewScore, decimals=3))))



    # win_prob = class_prob[:, 1] * 100
    win_prob = class_prob
    PredictedScore = np.around(win_prob, decimals=2)
    list_price = np.around(list_price, decimals=2)
    price_direction = ['None'] * len(list_price)
    for i in np.arange(len(list_price)):
        if (np.around(list_price[i], decimals=3) > np.around(customer_price, decimals=3)):
            price_direction[i] = "UP"
        else:
            if (np.around(list_price[i], decimals=3) < np.around(customer_price, decimals=3)):
                price_direction[i] = "DOWN"
            else:
                price_direction[i] = "KEEP"

    df_result = pd.DataFrame([], columns=['Direction', 'Price', 'PredictedScore'])
    df_result['Price'] = list_price
    # print([len(df_result),len(class_prob)])
    df_result['PredictedScore'] = PredictedScore
    df_result['Direction'] = price_direction
    if log_line == True:
        print("\n---------------------- Repricing Results ---------------------- \n")
        print(" Search from " + str(min(list_price)) + " to " + str(max(list_price)) + "\n")
    # df_result['PredictedScore'] = np.minimum.accumulate(df_result['PredictedScore'])
    df_result2 = df_result.sort_values('PredictedScore', ascending=False).head(10)[['Direction', 'Price', 'PredictedScore']]
    df_print = df_result2.copy()
    # df_print = df_result2.drop("PredictedScore",1)
    if log_line == True:
        print(df_print.to_string(index=False))

        if (type(winner_score).__name__ in ('list', 'tuple', 'ndarray')):
            print('Many Winner!')
            for eachscore in winner_score:
                print("==> PredictedScore(%) of current winner is: " + '{0:.2f}'.format(eachscore))
            print('\n')
        else:
            print("==> PredictedScore(%) of current winner is: " + '{0:.2f} \n'.format(winner_score))

    ## Recommended price based on the score and current price:
    # if(np.float(df_print.head(1)["PredictedScore"].values)>60):
    #     print("---------------------- RECOMMENDED PRICE: {0:.2f} ----------------------".format(np.float(df_print.head(1)["Price"])))
    # elif(np.float(df_print.head(1)["PredictedScore"].values)>40):
    #     if(np.float(df_print.head(1)["Price"]) > customer_price):
    #         print("---------------------- RECOMMENDED PRICE: {0:.2f} ----------------------".format(np.float(df_print.head(1)["Price"])))
    #     else:
    #         print("---------------------- RECOMMENDED PRICE: {0:.2f} ----------------------".format(customer_price))
    # else:
    #     print("---------------------- RECOMMENDED PRICE: {0:.2f} ----------------------".format(customer_price))

    ## Plot
    if (plotforus==True):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        # print(" \t--------------------------------------------------------------------------")
        sns.set_style("whitegrid", {'grid.linestyle': '--', 'axes.grid': True, 'grid.color': 'burlywood'})
        sample = df_result2
        controls = df_result
        ax1 = controls.plot(kind='area', x='Price', y='PredictedScore',  figsize=(10, 5),stacked=False,label="Score Area")
        ax2 = controls.plot(ax=ax1,kind='scatter', x='Price', y='PredictedScore', marker='.', c='b', s= 40, figsize=(10, 5) )
        ax2.set_xlim(ax1.get_xlim())
        ax3 = sample.plot(ax=ax1, kind='scatter', x='Price', y='PredictedScore', marker='d', c='r', s= 40, figsize=(10, 5),label="Top")
        ax3.set_xlim(ax1.get_xlim())
        # g = sns.pointplot(x=list_price, y=PredictedScore, color='r')
        # df_result.plot(kind='scatter', x="Price", y='PredictedScore', color='blue')
        ax1.set(ylabel='Winning Confidence Score', xlabel='Price')
        # start, end = ax1.get_xlim()
        # ax1.set_xticks(np.arange(start, end,abs(end-start)/10))
        ax1.set_xticks(df_result.Price)
        ax1.set_xticklabels(labels=df_result.Price, rotation=80)

        plt.vlines(x=customer_price, ymin=min(controls.PredictedScore), ymax=max(controls.PredictedScore), colors='LightSeaGreen', linestyles='dashed', label='Current: ' + str('{0:.2f}'.format(customer_price)))
        if(flag==True):
            plt.vlines(x=-9999, ymin=min(controls.PredictedScore), ymax=max(controls.PredictedScore), colors='DeepPink', linestyles='dashed',
                       label='Winner: ' + str("None"))
        else:
            plt.vlines(x=winpoint, ymin=min(controls.PredictedScore), ymax=max(controls.PredictedScore), colors='DeepPink', linestyles='dashed',
                       label='Winner: ' + str('{0:.2f}'.format(winpoint)))
        plt.legend()
        plt.tight_layout()
        # plt.text(10.1, 0, 'Current: ' + str(customer_price), rotation=90)
        plt.savefig(str(output_path) + "/" + str(nameXML) + '_' +str(name_clf) +'.png')
    if log_line == True:
        sys.stdout = saved
        fout.close()
        # plt.show(block=True)
    return df_result2
## UPDATE THE MODEL (NOT YET)
def updatemodel(olddf, newdf, mkID):
    frame = [olddf, newdf]
    df_update = pd.concat(frame)
    # mkID = df_update.MarketplaceId.unique()

    x_train, y_train = df_update.iloc[:, 1:].values, df_update.iloc[:, 0].values

    model_ = training(x_train, y_train)
    ## save model after training
    savemodel(model_, "./model", mkID)

    print("Model for market " + mkID + " is updated!")
    return model_
## REJECT THE OUTLIER IN DATA
def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]
## GET THE LIMITED PRICE TO RETURN THE FINAL PRICE LIST
def limited(list_price, customer_price, customer_status):
    final_list = np.sort(list_price)

    if (customer_status == -1):  ## lose
        if min(final_list) < customer_price:
            final_list = np.clip(final_list, min(final_list), customer_price)
        elif min(final_list) == customer_price:
            final_list = np.clip(final_list, customer_price, np.median(final_list))
        else:
            final_list = splitrange(customer_price, min(final_list), customer_price, len(final_list), customer_status)
    if (customer_status == 1):  ## Win
        if max(final_list) > customer_price:
            final_list = np.clip(final_list, customer_price, max(final_list))
        elif max(final_list) == customer_price:
            final_list = np.clip(final_list, np.median(final_list), customer_price)
        else:
            final_list = splitrange(max(final_list), customer_price, customer_price, len(final_list), customer_status)

    return final_list
## MAIN FUNCTION TO CREATE BINS OF RECOMENDED PRICE
def create_bins(minp, midp, maxp, fplot):
    if(minp==midp):
        final_arr = np.logspace(np.log10(midp), np.log10(maxp), 80, base=10.0)
        final_arr = np.array(np.unique(np.around(final_arr, decimals=2)))
    elif(midp==maxp):
        final_arr = np.logspace(np.log10(minp), np.log10(maxp), 50, base=10.0)
        final_arr = np.array(np.unique(np.around(final_arr, decimals=2)))
    else:
        arr1 = np.logspace(np.log10(midp), np.log10(maxp), 50, base=10.0)
        arr1 = np.unique(np.around(arr1, decimals=2))
        minpoint = min(arr1)
        arr2 = []
        for each in arr1:
            dif = each - minpoint
            tmp = minpoint - dif
            if (tmp >= minp) and (each != minp):
                arr2.append(tmp)
        arr2 = np.array(np.sort(arr2))
        final_arr = np.concatenate((arr2, arr1), axis=0)
        final_arr = np.array(np.unique(final_arr))
    if (fplot == True):
        plt.figure(figsize=(20, 8))
        plt.plot(final_arr, np.zeros(len(final_arr)), 'o')
        plt.plot(minp, 0, 'or')
        plt.plot(midp, 0, 'or')
        plt.plot(maxp, 0, 'or')
        plt.show()
    return final_arr
## MAIN FUNCTION FOR CREATING NEW SCORE
def decisionmaking(df):
    # probability = df["prob"]
    # FBA = df["IsFulfilledByAmazon"]
    # ideal = df["IdealPointCompetition"]
    df_ = df.copy()
    df_ = df_.reset_index(drop=True)
    ## OLD COMPUTATION
    df_["NewScore"] = df["IsFulfilledByAmazon"] * df["prob"] +  df["prob"]*df["IdealPointCompetition"] + df["prob"] * df["IsFeaturedMerchant"]

    # ## NEW COMPUTATION
    # df_["NewScore"] = minmaxscalerb(1/np.exp(df["IdealPointCompetition"])) + minmaxscalerb((df["prob"]))

    groups = df_.groupby(["CompetitionName"])
    df_["NewDecision"] = [0]*len(df_)
    df_["max"] = [0]*len(df_)

    for name, group in groups:
        gidx = group.index
        groupmax = max(group.NewScore)
        groupdec = ((group["NewScore"] == groupmax)-0.5)*2
        df_['NewDecision'].iloc[gidx] = groupdec
        # df_["max"].iloc[gidx] = [groupmax]*len(group)
        del groupmax, groupdec
    return df_
## GET BOUNDARY OF RECOMMENDED PRICES
def getboundary(min_threshold,winner_price,ten_percent_win_up,ten_percent_win_down,customer_price,ten_percent_cus_up,ten_percent_cus_down,customer_status):
    if customer_status == 1: ## win case
        lowerbound = max(winner_price,min_threshold)
        upperbound = ten_percent_win_up
        middlepoint = winner_price
    else: ## lose case
        if customer_price > winner_price:
            lowerbound = max(winner_price,min_threshold)
            upperbound = min(customer_price,ten_percent_win_up)
            middlepoint = winner_price
        elif customer_price < winner_price:
            lowerbound = max(ten_percent_cus_down, min_threshold)
            upperbound = winner_price
            middlepoint = winner_price
        else: ## equal
            lowerbound = max(ten_percent_cus_down,min_threshold)
            upperbound = winner_price
            middlepoint = winner_price

    return lowerbound, upperbound, middlepoint
## WRITE TO FILE WHILE PRINT TO THE SCREEN
class writer:
    def __init__(self, *writers):
        self.writers = writers

    def write(self, text):
        for w in self.writers:
            w.write(text)

import argparse, numpy as np

class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)
