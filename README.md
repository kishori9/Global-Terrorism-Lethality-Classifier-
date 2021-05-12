# Global-Terrorism-Lethality-Classifier-
Created a binary regressor to classify the attacks with and without casualties.

I have preprocessed data using pig using following steps:
1)	Registered piggybank and defined CSV loader. Loaded data into variable a using CSVLoader .
2)	Selected the column for modeling using FOREACH function. I have selected iyear, imonth, country, success, attacktype1 , weaptype1, scite3, and nkill
3)	Next, I selected all the records after year 2002 using FILTER.
4)	Converting nkill into categorical variable for binary regressor, to classify the records into casualty or no-casaulty.  
5)	Storing the preprocessed file with 95534 records on the machine.
6)	After preprocessing, I used Scala IDE for modeling. First, I imported the file to Scala IDE and created tuples. In the next step, dropped all the missing values in the dataset.
7)	I stored this data into a dataframe called terrorDF. 
8)	Used FILTER function to filter out records with iyear values as BR.
9)	Indexed all the columns to recode them as we can also used numeric data in the biary regressor.
10)	Used OneHotEncoder to encode categorical variables into numeric.
11)	Preprocessed scite3 column using TFIDF. 
12)	Used vector assembler to create a single feature out of all the x variables
13)	Next, used a pipeline to make data ready for modeling
14)	Modelled using logistic regressor
15)	Evaluated using precision, recall, f-score, accuracy and other metrics to get the output.

