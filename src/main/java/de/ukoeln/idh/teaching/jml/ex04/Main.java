package de.ukoeln.idh.teaching.jml.ex04;
import weka.classifiers.Classifier;
import weka.classifiers.misc.InputMappedClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.instance.RemoveDuplicates;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;



public class Main {

  public static void main(String[] args) throws Exception {
    
	  
	  Instances importedData = new Instances(new BufferedReader(new FileReader("src/main/resources/training.arff")));
	  int numberOfFeatures = importedData.numAttributes();
	  System.out.println(numberOfFeatures);
	  
	  
	  
	  Remove remove = new Remove();
	  remove.setAttributeIndices("3-6,8-25,28-35,37-47");
	  remove.setInputFormat(importedData);
	  
	  Instances purgedData = new Instances(Filter.useFilter(importedData, remove));
	  

	  StringToNominal stringConverter = new StringToNominal();
	  stringConverter.setInputFormat(purgedData);
	  String[] options = {"-R", "last"};
	  stringConverter.setOptions(options);
	  
	  Instances convertedData = new Instances(Filter.useFilter(purgedData, stringConverter));

	  
	  RemoveDuplicates removeDuplicates = new RemoveDuplicates();
	  removeDuplicates.setInputFormat(convertedData);
	  
	  Instances finalData = new Instances(Filter.useFilter(convertedData, removeDuplicates));
	  
	  if (finalData.classIndex() == -1){
		  finalData.setClassIndex(finalData.numAttributes()-1);
	       
	    }

	  
	  Classifier classifier = new InputMappedClassifier();
	  
	  
	  
	  Evaluation evaluation = new Evaluation(finalData);
	  evaluation.crossValidateModel(classifier, finalData, 10, new Random(1));
	  
	  System.out.println("Estimated Accuracy: " + evaluation.toSummaryString());
	  System.out.println(evaluation.toClassDetailsString());
	  System.out.println(evaluation.toMatrixString());
  }
  
  
  
}