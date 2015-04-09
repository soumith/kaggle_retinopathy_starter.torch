This folder showcases directly regressing from value 1.0 to 5.0, instead of treating 1,2,3,4,5 as separate classes.

The labels are normalized to be between -1.0 and 1.0 (instead of 1.0 to 5.0) by simply doing: newLabel = (oldLabel - 3.5) / 2.5
