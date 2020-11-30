package de.ukoeln.idh.teaching.jml.ex04;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
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
      
     



      Remove entfern = new Remove();
      entfern.setAttributeIndices("1,2,7,26,27,28,29");
      entfern.setInvertSelection(true);
      entfern.setInputFormat(importedData);

      Instances purgedData = new Instances(Filter.useFilter(importedData, entfern));

     

      StringToNominal stringConverter = new StringToNominal();
      stringConverter.setInputFormat(purgedData);
      String[] options = {"-R", "first-last"};
      stringConverter.setOptions(options);

      Instances convertedData = new Instances(Filter.useFilter(purgedData, stringConverter));

      


      RemoveDuplicates removeDuplicates = new RemoveDuplicates();
      removeDuplicates.setInputFormat(convertedData);

      Instances finalData = new Instances(Filter.useFilter(convertedData, removeDuplicates));

      if (finalData.classIndex() == -1){
          finalData.setClassIndex(finalData.numAttributes()-1);

        }


      Classifier refactor = new NaiveBayes();


      //Anwendung des Algorithmus
      Evaluation evaluation = new Evaluation(finalData);
      evaluation.crossValidateModel(refactor, finalData, 10, new Random(1));

      System.out.println("Estimated Accuracy: " + evaluation.toSummaryString());
      System.out.println(evaluation.toClassDetailsString());
      System.out.println(evaluation.toMatrixString());
  }



}