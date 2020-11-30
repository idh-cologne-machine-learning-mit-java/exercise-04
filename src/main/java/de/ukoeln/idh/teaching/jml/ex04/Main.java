package de.ukoeln.idh.teaching.jml.ex04;

import java.io.File;
import java.util.Random;

import com.lexicalscope.jewel.cli.CliFactory;
import com.lexicalscope.jewel.cli.Option;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.MergeInfrequentNominalValues;
import weka.filters.unsupervised.attribute.StringToNominal;

public class Main {

	public static void main(String[] args) throws Exception {
		// Parse command line options
		// This is done with the libarary JewelCLI
		// http://jewelcli.lexicalscope.com
		// Not necessary for the exercise, but useful
		Options options = CliFactory.parseArguments(Options.class, args);

		// load data set
		File inputFile = new File(options.getInput());
		ArffLoader loader = new ArffLoader();
		loader.setFile(inputFile);
		Instances instances = loader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);

		// Inititalize and use first filter
		StringToNominal filter0 = new StringToNominal();
		filter0.setAttributeRange("first-last");
		filter0.setInputFormat(instances);
		instances = Filter.useFilter(instances, filter0);

		// Inititalize and use second filter
		MergeInfrequentNominalValues filter1 = new MergeInfrequentNominalValues();
		filter1.setAttributeIndices("first-last");
		filter1.setMinimumFrequency(10);
		filter1.setInputFormat(instances);
		instances = Filter.useFilter(instances, filter1);

		// Initialise the classifier
		// If it has options, set them here
		NaiveBayes nb = new NaiveBayes();

		// Evaluation
		// Use the built-in handling of cross validation and evaluation
		Evaluation evaluation = new Evaluation(instances);
		evaluation.crossValidateModel(nb, instances, options.getNumberOfFolds(), new Random(options.getRandomSeed()));
		System.out.println(evaluation.toClassDetailsString());

		doCrossValidation(instances, nb, options);
	}

	/**
	 * Alternatively, if we want to know more about individual predictions (e.g.,
	 * for debugging), we can replicate the cross validation loop by ourselves,
	 * using the method below.
	 * 
	 * @param instances
	 * @param classifier
	 * @param options
	 * @throws Exception
	 */
	public static void doCrossValidation(Instances instances, Classifier classifier, Options options) throws Exception {

		// for each fold
		for (int f = 0; f < options.getNumberOfFolds(); f++) {
			System.err.print("Fold " + f + ": ");

			// split the data into train and test
			// (we don't have control over the random generator here)
			Instances train = instances.trainCV(options.getNumberOfFolds(), f);
			Instances test = instances.testCV(options.getNumberOfFolds(), f);

			// train classifier on training data
			classifier.buildClassifier(train);

			// Use Evaluation class to get predictions
			Evaluation eval = new Evaluation(test);
			double[] predictions = eval.evaluateModel(classifier, test);

			// print all or the first 100 (whichever is smaller) predictions
			// from this fold
			for (int i = 0; i < (Math.min(test.numInstances(), 100)); i++) {
				System.err.print(test.get(i).classValue());
				System.err.print(" ");
				System.err.print(predictions[i]);
				System.err.print("|");
			}
			System.err.println();
		}
	}

	public interface Options {
		@Option(defaultValue = "10")
		Integer getNumberOfFolds();

		@Option(defaultValue = "1")
		Integer getRandomSeed();

		@Option
		String getInput();

		@Option(defaultToNull = true)
		String getOutput();

	}
}