#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from os import listdir
from os.path import isfile, join
import re


def eda(datafiles, path_to_data):

    dfs = []
    file_dict = dict()
    for datafile in datafiles:
        xlsx = pd.ExcelFile(path_to_data + datafile)
        for n,sheet_name in enumerate(xlsx.sheet_names):
            name = datafile + f"_{n}"
            good_header = False
            header = 0
            while not good_header:
                df = pd.read_excel(xlsx, sheet_name, header=header)
                if any(col.startswith('Unnamed') for col in df.columns):
                    header += 1
                else:
                    good_header = True
                    dfs.append(df)
                    file_dict[name] = df

    for key, df in file_dict.items():
        nan_rows = df[df.isna().all(axis=1)]
        if len(nan_rows != 0):
            aux_df_list = []
            nan_indexes = nan_rows.index
            start_index = 0
            for index in nan_indexes:
                aux_df = df.iloc[start_index:index].reset_index(drop=True)
                if len(aux_df) > 1:
                    aux_df_list.append(aux_df)
                start_index = index
            aux_df = df.iloc[start_index:]
            aux_df_list.append(aux_df)
            file_dict[key] = aux_df_list

    new_df_list = []
    # Source 1
    df_source1_sheet_0 = file_dict["source1.xlsx_0"].iloc[0:-1] # remove last 
    new_df_source1_sheet_0 = pd.DataFrame(columns=["Article ID", "Weight", "Quantity", "Owner", "Grade", "Finish", "Thickness", "Width", "Description"])
    new_df_source1_sheet_0["Weight"] = df_source1_sheet_0["Gross weight (kg)"]
    new_df_source1_sheet_0["Description"] = df_source1_sheet_0["Description"]
    new_df_source1_sheet_0["Grade"] = df_source1_sheet_0["Grade"]
    new_df_source1_sheet_0["Finish"] = df_source1_sheet_0["Finish"]
    new_df_source1_sheet_0["Thickness"] = df_source1_sheet_0["Thickness (mm)"]
    new_df_source1_sheet_0["Width"] = df_source1_sheet_0["Width (mm)"]
    new_df_source1_sheet_0["Article ID"] = [str(x)+"/source1" for x in range(0,len(df_source1_sheet_0))] 
    new_df_source1_sheet_0["Owner"] = "source1"
    new_df_source1_sheet_0["Quantity"] = [1] * len(new_df_source1_sheet_0)
    new_df_list.append(new_df_source1_sheet_0)


    # Source 2
    df_source2_sheet_0_0 = file_dict["source2.xlsx_0"][0]
    df_source2_sheet_0_1 = file_dict["source2.xlsx_0"][1].iloc[2:] # remove firsrt 3 
    df_source2_sheet_0_2 = file_dict["source2.xlsx_0"][2].iloc[2:] # remove firsrt 3 
    df_source2_sheet_1_0 = file_dict["source2.xlsx_1"][0]
    df_source2_sheet_1_1 = file_dict["source2.xlsx_1"][1].iloc[3:] # remove firsrt 3
    df_source2_sheet_1_2 = file_dict["source2.xlsx_1"][2].iloc[3:] # remove firsrt 3 
    
    dfs_source2_1 = [df_source2_sheet_0_0, df_source2_sheet_0_1, df_source2_sheet_0_2]
    dfs_source2_2 = [df_source2_sheet_1_0, df_source2_sheet_1_1, df_source2_sheet_1_2]

    def extract_3_dimensions(material):
        dimensions = re.findall(r"(\d+[,|\.*\d*]*)\s*[m]*\s*[x|\*]\s*(\d+[,|\.]*\d*)\s*[m]*\s*[x|\*]*\s*(\d+[,|\.*\d*]*)\s*[m]*\s*", material)
        return dimensions[0] if dimensions else (0, 0)

    def extract_source2_features(material):
        other_features = re.sub(r"(\d+[,|\.*\d*]*)\s*[m]*\s*[x|\*]\s*(\d+[,|\.]*\d*)\s*[m]*\s*[x|\*]*\s*(\d+[,|\.*\d*]*)\s*[m]*\s*", "",material)
        features = other_features.split(" ")
        if len(features) == 2:
            return features[0], features[1], "-"
        else:
            return features[0], features[1], features[2]

    for df in dfs_source2_1:
        df = df.rename(columns={"Material ": "Material","Article ID ": "Article ID","weight ": "Weight"})
        new_df = df[["Article ID", "Weight"]]
        if "Quantity" in df.columns:
            new_df["Quantity"] = df["Quantity"]
        else:
            new_df["Quantity"] = [1] * len(df)
        new_df["Owner"] = "source2"
        new_df[["Height","Width", "Length"]] = df["Material"].apply(lambda x: pd.Series(extract_3_dimensions(x)))
        new_df["Length"] = new_df["Length"].str.replace(',', '.').astype(float)
        new_df["Height"] = new_df["Height"].str.replace(',', '.').astype(float)
        new_df["Width"] = new_df["Width"].str.replace(',', '.').astype(float)
        new_df[["Grade", "Coating", "Finish"]] = df["Material"].apply(lambda x: pd.Series(extract_source2_features(x)))
        new_df_list.append(new_df)
    for df in dfs_source2_2:
        df = df.rename(columns={"Material ": "Material","Article ID ": "Article ID","Weight ": "Weight", "Quantity ": "Quantity"})
        new_df = df[["Article ID", "Weight", "Quantity"]]
        new_df["Owner"] = "source2"
        new_df[["Height","Width", "Length"]] = df["Material"].apply(lambda x: pd.Series(extract_3_dimensions(x)))
        new_df["Length"] = new_df["Length"].str.replace(',', '.').astype(float)
        new_df["Height"] = new_df["Height"].str.replace(',', '.').astype(float)
        new_df["Width"] = new_df["Width"].str.replace(',', '.').astype(float)
        new_df[["Grade", "Coating", "Finish"]] = df["Material"].apply(lambda x: pd.Series(extract_source2_features(x)))
        new_df_list.append(new_df)


    # Source 3
    df_source3_sheet_0 = file_dict["source3.xlsx_0"]
    new_df_source3_sheet_0 = pd.DataFrame(columns=["Article ID", "Weight", "Quantity", "Owner", "Grade", "Finish", "Width", "Length", "Coating"])
    new_df_source3_sheet_0["Weight"] = df_source3_sheet_0["Libre"]
    new_df_source3_sheet_0["Article ID"] = (df_source3_sheet_0["Numéro de"]).astype(str) + "/" + (df_source3_sheet_0["Article"]).astype(str)
    new_df_source3_sheet_0["Owner"] = "source3"
    new_df_source3_sheet_0["Quantity"] = [1] * len(new_df_source3_sheet_0)

    def extract_2_dimensions(material):
        dimensions = re.findall(r"(\d+[,|\.*\d*]*)\s*[m]*\s*x\s*(\d+[,|\.]*\d*)\s*[m]*\s*", material)
        return dimensions[0] if dimensions else (None, None)

    def extract_source3_features(material):
        other_features = material.split(" ")
        return other_features[0], other_features[2], other_features[-2]

    new_df_source3_sheet_0[["Width", "Length"]] = df_source3_sheet_0["Matériel Desc#"].apply(lambda x: pd.Series(extract_2_dimensions(x)))
    new_df_source3_sheet_0["Length"] = new_df_source3_sheet_0["Length"].str.replace(',', '.').astype(float)
    new_df_source3_sheet_0["Width"] = new_df_source3_sheet_0["Width"].str.replace(',', '.').astype(float)
    new_df_source3_sheet_0[["Grade", "Coating", "Finish"]] = df_source3_sheet_0["Matériel Desc#"].apply(lambda x: pd.Series(extract_source3_features(x)))
    new_df_list.append(new_df_source3_sheet_0)


    # Last Cleaning
    for df in new_df_list:
        df.replace("VANILLA", 1, inplace=True) # only in 1 df in Quantity column
        df["Quantity"].fillna(1, inplace =True)
        df.fillna("-", inplace=True)
        df.replace(r'^\s*$', "-", regex=True, inplace=True)

    # remove columns with only zeros
    def remove_zero_columns(df):
        return df.loc[:, (df != 0).any(axis=0)]
    new_df_list = [remove_zero_columns(df) for df in new_df_list]

    # convert possible column dtypes to numbers
    for df in new_df_list:
        object_columns = df.select_dtypes(include='object').columns
        for col in object_columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
    
    return new_df_list

if __name__ == "__main__":

    path_to_data = "../resources/"
    onlyfiles = [f for f in listdir(path_to_data) if isfile(join(path_to_data, f))]
    datafiles = [f for f in onlyfiles if f.lower().endswith(".xlsx")]
    new_df_list = eda(datafiles, path_to_data)
    filenames = ["S1", "S2_0_0","S2_0_1","S2_0_2","S2_1_0","S2_1_1","S2_1_2","S3" ]
    print (new_df_list)
    
    for n,new_df in enumerate(new_df_list):
        new_df.to_csv(f"../resources/{filenames[n]}.tsv",sep="\t",index=False)

