package de.ukoeln.idh.teaching.jml.ex04;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.supervised.instance.ClassBalancer;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToNominal;

public class Main {

	public static void main(String[] args) throws IOException, Exception {
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("src/main/resources/training.arff"));
		
		Instances data = loader.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);

		// "all filters with all settings"	
		List<Filter> filters = new ArrayList<Filter>();
		
		Remove rmFilter = new Remove();
		rmFilter.setAttributeIndicesArray(new int[] { 1, 7, 23, 24, 25, 26, 47 });
		rmFilter.setInvertSelection(true);
		filters.add(rmFilter);
		
		StringToNominal strFilter = new StringToNominal();
		strFilter.setAttributeRange("first-last");
		filters.add(strFilter);
		
		ClassBalancer clsFilter = new ClassBalancer();
		filters.add(clsFilter);
		
		for (Filter filter : filters) {
			filter.setInputFormat(data);
			data = Filter.useFilter(data, filter);
		}
		// --------------------------------
		
		// "the classifier you've selected as the best"
		NaiveBayes classifier = new NaiveBayes();
		
		Evaluation e = new Evaluation(data);
		// "cross validation"
		e.crossValidateModel(classifier, data, 10, new Random(1));
		
		// "similar results as in the GUI"
			System.out.println(e.toSummaryString());
			/* "evaluation with precision/recall/f-score"
			 * 		optional:
			 * 			precision: e.precision(classIndex), e.weightedPrecision()
			 * 			recall: e.recall(classIndex), e.weightedRecall() 
			 * 			f-score: e.fMeasure(classIndex), e.weightedFMeasure() 
			 */
			System.out.println(e.toClassDetailsString());
			System.out.println(e.toMatrixString());
		// --------------------------------
	}

}