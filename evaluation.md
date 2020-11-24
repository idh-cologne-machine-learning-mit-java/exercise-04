# Evaluation of the Weka-GUI and the Weka-Java classifiers

Evaluation of both classifiers. They are both the same. One made with the Weka GUI and the other written in Java. Both with 10x cross validation.
They should be the same.


##Weka-GUI

Correctly Classified Instances       32221               92.4536 %
Incorrectly Classified Instances      2630                7.5464 %
Kappa statistic                          0.3586
Mean absolute error                      0.0266
Root mean squared error                  0.1389
Relative absolute error                 53.1366 %
Root relative squared error             87.9069 %
Total Number of Instances            34851

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0,996    0,737    0,929      0,996    0,962      0,456    0,843     0,973     O
                 0,362    0,002    0,844      0,362    0,506      0,546    0,929     0,539     B-EntityPER
                 0,185    0,003    0,710      0,185    0,293      0,352    0,753     0,283     I-EntityPER
                 0,026    0,000    0,417      0,026    0,049      0,102    0,898     0,187     B-EntityLOC
                 0,259    0,002    0,709      0,259    0,380      0,423    0,818     0,316     I-EntityLOC
                 0,000    0,000    ?          0,000    ?          ?        0,608     0,001     B-EntityWRK
                 0,061    0,000    0,400      0,061    0,105      0,155    0,727     0,038     I-EntityWRK
Weighted Avg.    0,925    0,669    ?          0,925    ?          ?        0,841     0,915

=== Confusion Matrix ===

     a     b     c     d     e     f     g   <-- classified as
 31485     9    65     5    44     0     2 |     a = O
   549   324    21     0     1     0     1 |     b = B-EntityPER
   984    46   237     0    14     0     0 |     c = I-EntityPER
   366     2     0    10     8     0     0 |     d = B-EntityLOC
   444     2    11     9   163     0     0 |     e = I-EntityLOC
    16     0     0     0     0     0     0 |     f = B-EntityWRK
    30     1     0     0     0     0     2 |     g = I-EntityWRK


##Weka-Java

Correctly Classified Instances       32221               92.4536 %
Incorrectly Classified Instances      2630                7.5464 %
Kappa statistic                          0.3586
Mean absolute error                      0.0266
Root mean squared error                  0.1389
Relative absolute error                 53.1366 %
Root relative squared error             87.9069 %
Total Number of Instances            34851

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0,996    0,737    0,929      0,996    0,962      0,456    0,843     0,973     O
                 0,362    0,002    0,844      0,362    0,506      0,546    0,929     0,539     B-EntityPER
                 0,185    0,003    0,710      0,185    0,293      0,352    0,753     0,283     I-EntityPER
                 0,026    0,000    0,417      0,026    0,049      0,102    0,898     0,187     B-EntityLOC
                 0,259    0,002    0,709      0,259    0,380      0,423    0,818     0,316     I-EntityLOC
                 0,000    0,000    ?          0,000    ?          ?        0,608     0,001     B-EntityWRK
                 0,061    0,000    0,400      0,061    0,105      0,155    0,727     0,038     I-EntityWRK
Weighted Avg.    0,925    0,669    ?          0,925    ?          ?        0,841     0,915

=== Confusion Matrix ===

     a     b     c     d     e     f     g   <-- classified as
 31485     9    65     5    44     0     2 |     a = O
   549   324    21     0     1     0     1 |     b = B-EntityPER
   984    46   237     0    14     0     0 |     c = I-EntityPER
   366     2     0    10     8     0     0 |     d = B-EntityLOC
   444     2    11     9   163     0     0 |     e = I-EntityLOC
    16     0     0     0     0     0     0 |     f = B-EntityWRK
    30     1     0     0     0     0     2 |     g = I-EntityWRK

##Summary
They are both the same.
