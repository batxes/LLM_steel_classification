# Steel Classification(LLM project)
![Steel](https://github.com/batxes/LLM_steel_classification/blob/main/steel_image.webp)

This is a LLM project that is able to classify types and features of steel from plain text. 
The project is an assignment that a company proposed for a Data Scientist position.

## Description of the project

A company works connecting steel providers and buyers. The steel providers have all their steel products and their features in excell files, and each excell file is different. The problem of this company is that it needs all the steel products in these excell files to automatically be cleaned and classified into a unified, standard table. 

This project is able to read these excell files, read each line containing different steel products and thei features, a classifying each of this features. For example, in our unified file we want to have for each product: 

 Type | quantity | size     | owner     
--- | --- | --- | --- 
 A12  | 500Kg    | 100x10x2 | Company A 

in one of the excell files from a random providers, we could have a line like this:

 Type | description                           | quantity  | product ID 
--- | --- | --- | --- 
 A12  | length: 100m, width: 50m, second hand | 500 units | ID: 1121   
  
The project reads this line uses regex or other ways of feature extraction and normalizes the data intoa unified data. 

Then, with unified data, a model is trained which is able to classify text into different features in the unified table. For example: The model reads "500Kg" and is able to classify the text into "Quantity".

The data are 3 excell files from 3 real companies.

## Instructions on how to run the project.
1) pipenv install -r requirements.txt
2) python -W ignore run.py

## Instructions on how to test the model.
1) python scripts/predict.py

## The standardized dataset in CSV format.
resources/union.tsv

## Report summarizing the approach and results.

I first created a jupyter notebook and started exploring the data. I saw that the 3 files are different and contain different variables. I also saw that one of the excel files contains 2 tabs and multiple tables in each tab. I thought on going through the 3 files and saving them into dataframes, and if the file contained tabs, going through each tab. If one of the dataframes contained lines of NaNs, indicated that another table was starting, so I also made sure to save those as independent dataframes.


Then, I tried to save as many variables as possible, and I also extracted different dimensions if they were embedded in the description, using regex.


Finally I had to do some hardcoded cleaning, like removing out-of-context text from the quantity column.


I then prepared the data for training. I concatenated all the dataframes into one. Put all variables as categories and their values as text. Then I split the data into train, validation and test (60, 20, 20). I used the transformers library to work with pre-trained BERT models. I also applied a step to evaluate the model. Then I save the model. In the test script I hard coded a couple of steel properties to predict the category.

These are the metrics of the evaluation of the model (not deterministic):

eval_loss |  eval_Accuracy |   eval_F1 |  eval_Precision |  eval_Recall 
--- |  --- | --- | --- | --- | 
 train  |  0.679016 |       0.734959 |  0.714131 |        0.883501 |     0.734953  
 val    |  0.741721 |       0.712895 |  0.681331 |        0.786165 |     0.714636  
 test   |  0.738569 |       0.722628 |  0.685436 |        0.766949 |     0.720938  

and the prediction result:

`ungebeizt, nicht geglüht -> [{'label': 'Finish', 'score': 0.9299754500389099}]`

`C100S -> [{'label': 'Grade', 'score': 0.9502202272415161}]`

## Plan for the future expansion of the algorithm and the potential most difficult challenges.

There could be many improvements. First, the EDA needs to be refined. I would remove automatically all the lines containing NaNs in the "Article ID" for example. I would also check for columns that contain strings and not numbers in variables that need to be numbers and remove. The code should be modified so it accepts one file at a time, so it is scalable. Right now it takes all 3 together.

The final variables that are used to concatenate all dataframes should also be standardized. Maybe, the variables that do not appear in some data files of different providers could be appended to an auxiliary column. LIke Provider 1 contains "description", which provider 2 and 3 do not. It could also be parsed to try to find features that could be applied to other variables.

The tokenization could also be improved. RIght now, the numbers are converted into strings, which is not ideal.

I missed the final step which would be to recreate the unified dataset. For that, I would fed the model with each of the features/variables of each individual steel product and append to a new table, in the category that the model predicts. Ideally, this would have to be done only for the variables that are not automatically set in a previous step. For example, if we get a product with a column called "important" and below we have values like "GXES", the model would predict these values to belong to the coating column. Or if we get column names in Chinese, only knowing the value of these columns, the model should be able to create a database with the new predicted columns.


## Tools/tech used:

<p align="center">
    <img src="icons/pytorch.webp" width="200">
    <img src="icons/transformers.svg" width="200">
    <img src="icons/scikit.png" width="150">
    <img src="icons/pandas.webp" width="150">
</p>



