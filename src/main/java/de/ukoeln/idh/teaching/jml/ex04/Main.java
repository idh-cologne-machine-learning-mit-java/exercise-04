package de.ukoeln.idh.teaching.jml.ex04;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class Main {
  public static void main(String[] args) throws IOException, Exception {

    // load data
    ArffLoader loader = new ArffLoader();
    loader.setFile(new File("src/main/resources/training.arff"));
    Instances instances = loader.getDataSet();
    instances.setClassIndex(instances.numAttributes() - 1);

    // remove attributes
    Remove removeAttributes = new Remove();
    removeAttributes.setAttributeIndices("1, 2, 5, 12, 13, 16, 17, 20, 22, 26, 27, 29, 30");
    removeAttributes.setInvertSelection(true);
    removeAttributes.setInputFormat(instances);
    instances = Filter.useFilter(instances, removeAttributes);

    // convert string attributes to nominal
    StringToNominal stringToNominal = new StringToNominal();
    stringToNominal.setAttributeRange("first-last");
    stringToNominal.setInputFormat(instances);
    instances = Filter.useFilter(instances, stringToNominal);

    // train classifier
    Classifier nbClassifier = new NaiveBayes();
    Evaluation evaluation = new Evaluation(instances);
    evaluation.crossValidateModel(nbClassifier, instances, 10, new Random(1));

    System.out.println(evaluation.toMatrixString());
    }


}