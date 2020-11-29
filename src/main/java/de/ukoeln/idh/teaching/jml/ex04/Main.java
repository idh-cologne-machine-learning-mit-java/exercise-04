package de.ukoeln.idh.teaching.jml.ex04;

import java.io.File;
import java.util.Random;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.MergeTwoValues;
import weka.filters.supervised.instance.Resample;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomTree;

public class Main {
  public static void main(String[] args) throws IOException, Exception {
	  // load training data
	  ArffLoader loader = new ArffLoader();
	  loader.setFile(new File("src/main/resources/training.arff"));
	  Instances instances = loader.getDataSet();
	  
	  //
	  // filter steps
	  //
	  
	  // remove attributes
	  Remove rmFilter = new Remove();
	  rmFilter.setAttributeIndices("2, 6, 16, 18, 20, 22, 26, 27, 29, 30, 48");
	  rmFilter.setInvertSelection(true);
	  rmFilter.setInputFormat(instances);
	  instances = Filter.useFilter(instances, rmFilter);
	  
	  // string to nominal
	  StringToNominal s2nFilter = new StringToNominal();
	  s2nFilter.setAttributeRange("first-last");
	  s2nFilter.setInputFormat(instances);
	  instances = Filter.useFilter(instances, s2nFilter);
	  
	  // merge B/I-Attributes of same NE tag
	  MergeTwoValues mergeValuesFilter = new MergeTwoValues();
	  int[] mergeFirstIndices = { 2, 3, 4 };
	  int[] mergeSecondIndices = { 3, 4, 5 };
	  mergeValuesFilter.setAttributeIndex("last");
	  for (int i = 0; i < mergeFirstIndices.length; i++) {
		  mergeValuesFilter.setFirstValueIndex(Integer.toString(mergeFirstIndices[i]));
		  mergeValuesFilter.setSecondValueIndex(Integer.toString(mergeSecondIndices[i]));
		  mergeValuesFilter.setInputFormat(instances);
		  instances = Filter.useFilter(instances, mergeValuesFilter);
	  }
	  
	  instances.setClassIndex(instances.numAttributes() - 1);
	  
	  // oversample minority classes
	  Resample resampleFilter = new Resample();
	  resampleFilter.setBiasToUniformClass(1.0);
	  resampleFilter.setSampleSizePercent(180);
	  resampleFilter.setInputFormat(instances);
	  instances = Filter.useFilter(instances, resampleFilter);
	  
	  //
	  // model training
	  //
	  
	  RandomTree tree = new RandomTree();
	  Evaluation eval = new Evaluation(instances);
	  eval.crossValidateModel(tree, instances, 10, new Random(1));
	  
	  System.out.println(eval.toSummaryString());
	  System.out.println(eval.toMatrixString());
	  

	  
	  
	  
	  
	  
  }
  
}