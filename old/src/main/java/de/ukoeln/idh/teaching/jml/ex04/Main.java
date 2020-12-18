package de.ukoeln.idh.teaching.jml.ex04;

import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.MergeInfrequentNominalValues;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.core.Instances;
import weka.core.Instance;

import java.util.Random;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;

public class Main {
  public static void main(String[] args) throws Exception {

    // loading data
    DataSource source = new DataSource("src/main/resources/training.arff");
    Instances data = source.getDataSet();
    if (data.classIndex() == -1) {
      data.setClassIndex(data.numAttributes() - 1);
    }

    // wanted to select with AttrbiuteSelection but does not allow selection via
    // indices
    // cosulted
    // https://www.researchgate.net/post/How_can_we_select_specific_attributes_using_WEKA_API
    // for help
    // used Remove with invertedSelection set to "true" -> keep selected, remove
    // unselected (bc. noOfAttributesKept < noOfAttrbiutesRemoved)
    Remove toKeep = new Remove();
    toKeep.setAttributeIndices("first,2,6,7,11,26,27,36,last");
    toKeep.setInvertSelection(true);
    toKeep.setInputFormat(data);
    Instances filtered = Filter.useFilter(data, toKeep);

    // conversion String to Nominal
    StringToNominal convertToNominal = new StringToNominal();
    convertToNominal.setAttributeRange("first-last");
    convertToNominal.setInputFormat(filtered);
    Instances preprocess0 = Filter.useFilter(filtered, convertToNominal);

    // just a test to count number of values
    // thought I could use this for RemoveWithValue
    // discarded

    /*
     * int i=0; int[] countAttributes = new
     * int[preprocess0.attribute(i).numValues()]; for (Instance instance :
     * preprocess0) { countAttributes[(int) instance.value(i)]++; }
     * System.out.println(countAttributes.length); for(int i=0; i <
     * countAttributes.length; i++){ if(countAttributes[i]>1){
     * System.out.println(countAttributes[i]); } }
     */

    //
    // removing infrequent values of "word" attribute
    // MergeInfrequentNominalValues bc couldn't find the parameters to remove
    // infrequent values using RemoveFrequentValues (API states that this method
    // does either/or -> doesn't work)
    //

    MergeInfrequentNominalValues cleaner = new MergeInfrequentNominalValues();
    String[] options = { "-N", "5", "-R", "first-last" };
    cleaner.setOptions(options);
    cleaner.setInputFormat(preprocess0);
    Instances preprocessFinal = Filter.useFilter(preprocess0, cleaner);

    // in the GUI, I had to use multinomialText, only executing on String
    // attributes,
    // though all preprocessing steps converting to nominal have been selected
    // analogously to this code base?!
    // there simply was no option to choose NB (& I don't get why?!)
    NaiveBayes naba = new NaiveBayes();

    // evaluate using built-in
    Evaluation eval = new Evaluation(preprocessFinal);
    eval.crossValidateModel(naba, preprocessFinal, 5, new Random(1));

    // output results
    System.out.println("Metrics: " + eval.toSummaryString());

    // evaluate

    // cross-validate

    // classify

  }

}