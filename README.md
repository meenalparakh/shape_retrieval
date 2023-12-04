# 3D Objects Retrieval using LSH

## Next steps: 
- __read both the code and the lecture notes__: *look into what the LSH library is doing (done) - lay it out as a pseudocode and compare it with what is given in the lecture notes*

- __half done__ - *compare what does sparse matrix do, and if needed implement your variant as well - answer - useful if the key representation is binary - but not for projection - as my feature vector are not sparse - but need to look into how sparse they are*.

- __done__ - *start writing code as per the lecture notes and everything that you can think of - if there is some parameter you don't know how to set - make it available throught the class argument.*


- steps for how to visualize the mapping
take the tables

- code for the following
   - visualization of the hash table - where does a point gets mapped to in the hash table 
   - visualization both of the feature vector and low dimensional hash values (before binning).
   - large scale hashing of the dataset and retrieval of objects

- think about how to represent the different parameter variations

- seems like peaks are important in the graphs - how to capture those in the representation, because distance functions care more about overall shape.


- MUST DO:
BASELINE - HOW DOES LSH COMPARE WITH NAIVE APPROACH THEORETICALLY & EXPERIMENTALLY IN TERMS OF RUNTIME AND COMPUTE REQUIRED AMMORTIZED 

- use PCA to plot the different groups - not sure if this will be useful
    but there is no harm in trying it out
- things to try - think of ways to visualize how different hyperparams 
    affect the object retrieval
- things to try - faster ways of computing histograms (complexity wise)
- things to try - different representations, espescially learned through 
    self-supervision on a subset of data
- things to pay attention to - try to search for new algorithms and
    add them to the implementation

