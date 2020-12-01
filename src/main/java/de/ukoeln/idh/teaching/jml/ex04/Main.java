package de.ukoeln.idh.teaching.jml.ex04;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.Filter;
import weka.filters.supervised.instance.ClassBalancer;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToNominal;

public class Main {
	public static void main(String[] args) throws Exception {
		// Trainingsdaten einlesen
		BufferedReader reader = new BufferedReader(new FileReader("src/main/resources/training.arff"));
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();
		data.setClassIndex(data.numAttributes() - 1);
		
		// Filter definieren ...
		List<Filter> list = new ArrayList<Filter>();      
        Remove r = new Remove();
        r.setAttributeIndicesArray(new int[] {1,7,23,24,25,26,35,47});
        r.setInvertSelection(true);
        list.add(r);
        
        StringToNominal s = new StringToNominal();
        s.setAttributeRange("first-last");
        list.add(s);
        
        list.add(new ClassBalancer());
        
        // ... und anwenden
        for (Filter filter : list) {
            filter.setInputFormat(data);
            data = Filter.useFilter(data, filter);
        }
		
		// Naive Bayes Classifier evaluieren (Kreuzvalidierung)
		NaiveBayes classifier = new NaiveBayes();		
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(classifier, data, 10, new Random(15));
		
		// Zusammenfassung der Evaluationsergebnisse
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toClassDetailsString());
		
		

	}

}