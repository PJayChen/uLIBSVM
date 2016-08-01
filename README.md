A modified svm_predict of [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) for microcontroller.

This project will translate trained svm model file into c struct declaration.

Only support for classification.

Prerequisites
-------------
* A SVM model trained by svm_train of LIBSVM
* [Optional] A SVM scale range file of LIBSVM

Building
-------------------------
* Compile the model translation program
```
gcc modelFile2cStruct -o modelFile2cStruct
```
Translate trained svm model
-------------------------
```
./modelFile2cStruct model_file > output_file
```
* The translated svm model struct declaration will store in output_file.
* After that, you are able to add the contain in output_file into your program.
* An example
```
./modelFile2cStruct ./data/templates.tf.model > model
```

Example
-------------------------
* The required function for svm prediction is modified and an usage example is in main.c.
* Step
    - Copy translated svm model into your program.
    - Copy the necessary function in main.c into your program.
    - Include the simplified svm.h into your program.
* note: the scale function is optional. 
    - If you need scale your feature vector, you need modified following declaration,
    - double lower=xxx,upper=xxx;
    - double feature_max[] = {xxx};
    - double feature_min[] = {xxx};

