package de.ukoeln.idh.teaching.jml.ex04;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.Filter;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import java.util.Random;

public class Main {
  public static void main(String[] args) throws Exception {
	  Instances data = loadAndPreprocess("src/main/resources/training.arff");
	  
	  AbstractClassifier classifier = model();
	  
	  Evaluation eval = new Evaluation(data);
	  eval.crossValidateModel(classifier, data, 10, new Random(1));
	 
	  //Print the evaluation
	  System.out.println(eval.toSummaryString());
	  System.out.println(eval.toClassDetailsString());
	  System.out.println(eval.toMatrixString());
  }
  
  //Gets a Classifier Model for Evaluation
  //Using AbstractClassifier for easier swapping of models
  public static AbstractClassifier model() throws Exception {
	  //Using randomForest as the base classifier with 2 trees
	  RandomForest forest = new RandomForest();
	  forest.setNumIterations(2);
	  
	  //Setting a CostMatrix, because the data is skewed
	  String matrixString = "[0 5 5 5 5 5 5; "
	  						+ "5 0 1 1 1 1 1; "
	  						+ "5 1 0 1 1 1 1; "
	  						+ "5 1 1 0 1 1 1; "
	  						+ "5 1 1 1 0 1 1; "
	  						+ "5 1 1 1 1 0 1; "
	  						+ "5 1 1 1 1 1 0]";
	  
	  CostMatrix matrix = CostMatrix.parseMatlab(matrixString);
	  
	  //Using a CostSensitiveClassifier with randomForest
	  CostSensitiveClassifier classifier = new CostSensitiveClassifier();
	  classifier.setCostMatrix(matrix);
	  classifier.setClassifier(forest);

	  return classifier;
  }
  
  //Loads the .arff file and preprocesses it
  public static Instances loadAndPreprocess(String file) throws Exception {
	  DataSource loader = new DataSource(file);
	  Instances data = loader.getDataSet();
	  data.setClassIndex(data.numAttributes() - 1);
	  for(int i = data.numAttributes() - 2; i >= 2; i--){
		  data.deleteAttributeAt(i);
	  }
	  
	  StringToNominal converter = new StringToNominal();
	  String range = "first-last";
	  converter.setAttributeRange(range);
	  converter.setInputFormat(data);
	  
	  Instances filteredData = Filter.useFilter(data, converter);
	  return filteredData;
  }
  
}