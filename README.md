#CONTRIBUTERS#

- Sverri Lyager Jacobsen (fgv469)
- Christoph Friedemann Ueberfuhr (vsn354)
- Oskar Jørgensen (hln323)


#INSTALLATION AND SETUP#

To properly setup up the project please ensure that you have placed the following files in main folder (gds123):  

- 995.000_rows.csv
    A link to the download and more information about the dataset can be found here: https://github.com/several27/FakeNewsCorpus. Please note that the file must be named "995.000rows" in order to work.

- LIAR Dataset

    This dataset can be downloaded via https://www.cs.ucsb.edu/~william/data/liar_dataset.zip. 
    It is important that the dataset is unzipped.


- bbc_articles_content.json
    
    The scraped articles from assignment 2  can be generated or replaced by the code we handed in for assignment 2. The current code here processes .json scraped article files even though it's slow. The reasoning for this is that we didn't succeed in scraping metadata to csv.

#REQUIREMENTS#

Please ensure that you have installed the following libraries in you python enviroment: 

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
- joblib


#RUNNING THE PROJECT#

The code currently exists as notebooks. To run the code we therefore recommend running the following files in an environment that can run notebooks:

- Part1.ipynb
- Part2.ipynb
- Part3.ipynb
- Part4.ipynb

That should produce everything necessary to fulfill the task description. Some of the later parts require files that are generated by some of the older parts so make sure to run everything in order (1 to 4)

#NOTES#

Some parameters (such as sample size) can be adjusted but are currently set to the values that the report refers to.
