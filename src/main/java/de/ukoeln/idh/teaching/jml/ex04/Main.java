package de.ukoeln.idh.teaching.jml.ex04;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;


public class Main {

	public static void main(String[] args) throws IOException, Exception {
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("src/main/resources/training.arff"));
		Instances inst = loader.getDataSet();
		inst.setClassIndex(inst.numAttributes() - 1);

		List<Filter> allFilters = new ArrayList<Filter>();
		
		Remove rmFilter = new Remove();
	  rmFilter.setAttributeIndices("1, 2, 6, 7, 16, 18, 26, 27, 48");
		rmFilter.setInvertSelection(true);
		allFilters.add(rmFilter);
		
		StringToNominal stnFilter = new StringToNominal();
		stnFilter.setAttributeRange("first-last");
		allFilters.add(stnFilter);
		
		for (Filter filter : allFilters) {
			filter.setInputFormat(inst);
			inst = Filter.useFilter(inst, filter);
		}
    
    J48 tree = new J48();
    String[] options = new String[1];
    options[0] = "-U"; 
    tree.setOptions(options);
    tree.buildClassifier(inst);

		Evaluation evaluation = new Evaluation(inst);
		evaluation.crossValidateModel(tree, inst, 10, new Random(1));
		
		System.out.println(evaluation.toSummaryString());
		System.out.println(evaluation.toClassDetailsString());
		System.out.println(evaluation.toMatrixString());

	}

}