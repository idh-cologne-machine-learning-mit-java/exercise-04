package de.ukoeln.idh.teaching.jml.ex04;

public class Main {
  public static void main(String[] args) {
    SimpleAnalyzer sa = new SimpleAnalyzer();
    sa.autoProcess("src/main/resources/training.arff");
  }
  }