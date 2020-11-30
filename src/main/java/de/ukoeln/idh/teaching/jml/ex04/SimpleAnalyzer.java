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
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;


public class SimpleAnalyzer {
    /**
     *
     * @param path the path to the File which contains arff data
     * @return a Data Instances instance based on the arff file
     * @throws IOException
     */
    public Instances load(String path) throws IOException {
        FileReader file = new FileReader(path);
        BufferedReader data = new BufferedReader(file);
        return new Instances(data);
    }

    /**
     *
     * @param data
     * @param features An array of features that are supposed to be removed
     * @return a Data Instances instance based the data where features are removed
     */
    public Instances remove(Instances data, int[] features) {
        Remove remove = new Remove();
        String attributes = String.join(
                ",",
                Arrays.stream(features)
                    .mapToObj(String::valueOf) // map values to string object
                        .toArray(String[]::new) // create an string[] from map ;
        );
        remove.setAttributeIndices(attributes);
        remove.setInvertSelection(true); // revert
        try {
            remove.setInputFormat(data);
            return new Instances(Filter.useFilter(data, remove));
        } catch (Exception e) {
            System.out.println("Could not remove features: " + features);
            return data;
        }
    }

    /**
     *
     * @param data
     * @return a Data Instances instance based the data where string features are now nominal
     */
    public Instances toNominal(Instances data) {
        StringToNominal stringConverter = new StringToNominal();
        try {
            stringConverter.setInputFormat(data);
            String[] options = {"-R", "first-last"};
            stringConverter.setOptions(options);
            return new Instances(Filter.useFilter(data, stringConverter));
        } catch (Exception e) {
            System.out.println("Could not make features nominal" );
            return data;
        }
    }

    public Instances clearData(Instances data) {
        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        try {
            removeDuplicates.setInputFormat(data);
            return new Instances(Filter.useFilter(data, removeDuplicates));
        } catch (Exception e) {
            System.out.println("Could not clea data from duplicates" );
            return data;
        }
    }

    /**
     *
     * @param data the data instance which is supposed to be evaluated
     * @param targetIndex the target feature of the dataset
     * @param classifier the classifier which is supposed to be taken
     */
    public void evaluate(Instances data, int targetIndex, Classifier classifier)  {
        try {
            data.setClassIndex(targetIndex);
            Evaluation evaluation = new Evaluation(data);
            evaluation.crossValidateModel(classifier, data, 10, new Random(1));

            System.out.println("Estimated Accuracy: " + evaluation.toSummaryString());
            System.out.println(evaluation.toClassDetailsString());
            System.out.println(evaluation.toMatrixString());
        } catch (Exception e) {
            System.err.println("Could not evaluate with classifier " + classifier.toString() + " and targetIndex" + targetIndex);
            e.printStackTrace();
        }

    }

    /**
     * evaluates with NaiveBayes classifier and the last attribute as target
     * @param data the data instance which is supposed to be evaluated
     */
    public void evaluate(Instances data)  {
        this.evaluate(
                data,
                data.numAttributes() - 1, // last one
                new NaiveBayes()
        );
    }

    /**
     * does everything with defaults
     * @param path to arff
     */
    public void autoProcess(String path) {
        Instances data;
        try {
           data = this.load(path);
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        data = this.remove(data, new int[] { 1,2,7,26,27,28,29 });
        data = this.toNominal(data);
        data = this.clearData(data);
        this.evaluate(data);
    }
}