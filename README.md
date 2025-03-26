This GDS exam project code can be run once you have the 995,000_rows.csv file in the gds123 folder (main folder)

The code currently exists as .py files and notebooks, where the notebooks are an attempt to polish the code from the .py files and make everything more concise. To run the code we therefore recommend running the following files:
Part1.ipynb
Part2.ipynb
Part3.ipynb
Part4.ipynb

That should produce everything necessary to fulfill the task description. Some of the later parts require files that are generated by some of the older parts so make sure to run everything in order (1 to 4)

Some parameters (such as sample size) can be adjusted but are currently set to the values that the report refers to.

The scraped articles from assignment 2 (bbc_articles_content.json) can be generated or replaced by the code we handed in for assignment 2. The current code here processes .json scraped article files even though it's slow. The reasoning for this is that we didn't succeed in scraping metadata to csv.