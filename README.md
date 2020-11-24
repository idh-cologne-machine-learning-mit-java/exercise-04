# Exercise 4: Weka as Java API


The goal of the second exercise is to implement what you did for last exercise with a GUI in the form of Java code.

## Step 1
Please `clone` the repository `https://github.com/idh-cologne-machine-learning-mit-java/exercise-04`.

Create a new branch, using your UzK username.

## Step 2
The `pom.xml` file already contains Weka as a dependency, such that you can directly start coding. Please implement the workflow you have been using on the training data in exercise 3 as exactly as possible. This includes:

- all filters with all settings
- cross validation
- evaluation with precision/recall/f-score
- the classifier you've selected as the best

Verify that your code produces similar results as in the GUI. 

The javadoc for Weka 3.8.4 can be found [here](https://javadoc.io/doc/nz.ac.waikato.cms.weka/weka-stable/latest/index.html). 

Last week's training data has been added to the `src/main/resources`-folder. 

## Step 3
Commit your changes and push them to the server.