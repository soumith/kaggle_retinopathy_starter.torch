Some insights on the dataset.

Total Classes: 5

Total Examples: 35126

```Labels: {'0','1','2','3','4'}```

Class  | #Examples
------ | ---------
'0'    |  25810
'1'    |   2443
'2'    |   5292
'3'    |    873
'4'    |    708


Deviation in Label | #Examples with same Id
------------------ | ----------------------
 0                 |  15323
 1                 |   1485
 2                 |    723
 3                 |     11
 4                 |     21


Ids that differ by ```>= 2``` are the interesting ones.
Ids that differ by ```4```:

```
['16920','37438','28743','35199','30725','7821','28976',
 '42442','11807','36125','19005','14765','13938','29058',
 '3611','10321','36531','30570','11730','3501','28524']
 ```
 It'll be interesting to look at these examples to see how they differ, If there's a way to see the difference in their feature activations then we can come up with a good way to generalize (sort of, not guaranteed, but doesn't hurt to try). 
 
 Also, metric learning via siamese is justified here a few classes have very less number of examples to make the network learn anything about them.
