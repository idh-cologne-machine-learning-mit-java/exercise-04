package de.ukoeln.idh.teaching.jml.ex04;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.filters.Filter;
import weka.filters.supervised.instance.ClassBalancer;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.Remove;


public class Main {
  public static void main(String[] args) throws IOException, Exception {
	//load data
    ArffLoader loader = new ArffLoader();
    loader.setFile(new File("src/main/resources/training.arff"));
    Instances instances = loader.getDataSet();
    
    //remove irrelevant attributes (inverted remove)
    Remove rmFilter = new Remove();
    rmFilter.setAttributeIndices("2, 6, 7, 12, 13, 26, 27, 48");
    rmFilter.setInvertSelection(true);
    rmFilter.setInputFormat(instances);
    instances = Filter.useFilter(instances, rmFilter);
    
    //set class index
    instances.setClassIndex(instances.numAttributes() - 1);
    
    //convert string to nominal
    StringToNominal stmFilter = new StringToNominal();
    stmFilter.setAttributeRange("1, 3, 6, 7");
    stmFilter.setInputFormat(instances);
    instances = Filter.useFilter(instances, stmFilter);
    
    //use class balancer
    ClassBalancer balancer = new ClassBalancer();
    balancer.setNumIntervals(10);
    balancer.setInputFormat(instances);
    instances = Filter.useFilter(instances, balancer);
    
    //train naive bayes
    NaiveBayes bayesClf = new NaiveBayes();
    bayesClf.setBatchSize("100");
    bayesClf.setNumDecimalPlaces(2);
    
    //evaluate
    Evaluation eval = new Evaluation(instances);
    eval.crossValidateModel(bayesClf, instances, 10, new Random(1));
    
    //print results
    System.out.println(eval.toSummaryString());
    System.out.println(eval.toClassDetailsString());
    System.out.println(eval.toMatrixString());
    
  }
  
}