package de.ukoeln.idh.teaching.jml.ex04;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomialText;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.instance.RemoveDuplicates;

public class Main {
	
  public static void main(String[] args) throws Exception {
    
	  // load training-data
	  Instances importedData = new Instances(new BufferedReader(new FileReader("src/main/resources/training.arff")));
	  int numberOfFeatures = importedData.numAttributes();
	  System.out.println(numberOfFeatures);
	  
	  Instances purgedData = new Instances(Filter.useFilter(importedData, null));
	  
	  StringToNominal stringConverter = new StringToNominal();
	  stringConverter.setInputFormat(purgedData);
	  String[] options = {"-R", "first-last"};
	  stringConverter.setOptions(options);
	  
	  Instances convertedData = new Instances(Filter.useFilter(purgedData, stringConverter));
	  
	  RemoveDuplicates removeDuplicates = new RemoveDuplicates();
	  removeDuplicates.setInputFormat(convertedData);
	  
	  Instances finalData = new Instances(Filter.useFilter(convertedData, removeDuplicates));
	  
	  // naive bayes multinominal text
	  Classifier classifier = new NaiveBayesMultinomialText();
	  
	  // do the evaluation here
	  Evaluation evaluation = new Evaluation(finalData);
	  evaluation.crossValidateModel(classifier, finalData, 1, new Random(1));
	  
	  // output of values (MatrixString and other detailed values)
	  System.out.println("Estimated Values: " + evaluation.toSummaryString());
	  System.out.println(evaluation.toClassDetailsString());
	  System.out.println(evaluation.toMatrixString());
	  	  
  }
  
}