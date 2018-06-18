# Indian Sign Language Recognition

Sign Languages are a set of languages that use predefined actions and movements to convey a message. These languages are primarily developed to aid deaf and other verbally challenged people. They use a simultaneous and precise combination of movement of hands, orientation of hands, hand shapes etc. Different regions have different sign languages like American Sign Language, Indian Sign Language etc. We focus on Indian Sign language in this project.

Indian Sign Language (ISL) is a sign language that is predominantly used in South Asian countries. It is sometimes referred to as Indo-Pakistani Sign Language (IPSL). There are many special features present in ISL that distinguish it from other Sign Languages. Features like Number Signs, Family Relationship, use of space etc. are crucial features of ISL. Also, ISL does not have any temporal inflection.

In this project, we aim towards analyzing and recognizing various alphabets from a database of sign images. Database consists of various images with each image clicked in different light condition with different hand orientation. With such a divergent data set, we are able to train our system to good levels and thus obtain good results.

We investigate different machine learning techniques like:
- [K-Nearest-Neighbours](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine)
- [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network)

## Getting Started
### Prerequisites
Before running this project, make sure you have following dependencies - 
* [Dataset](https://drive.google.com/file/d/15bikHgG8Y13vWdMQ-6-MK0y8AI3sGBfe/view?usp=sharing) (Download the images from this link)
* [Python 3.6](https://www.python.org/downloads/)
* [pip](https://pypi.python.org/pypi/pip)
* [OpenCV](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html)

Now, using ```pip install``` command, include following dependencies 
+ Numpy 
+ Pandas
+ Sklearn
+ Scipy
+ Opencv
+ Tensorflow

### Running
To run the project, perform following steps -

 1. Put all the training and testing images in a directory and update their paths in the config file *`common/config.py`*.
 2. Generate image-vs-label mapping for all the training images - `generate_images_labels.py train`.
 3. Apply the image-transformation algorithms to the training images - `transform_images.py`.
 4. Train the model(KNN & SVM) - `train_model.py <model-name>`. Note that the repo already includes pre-trained models for some algorithms serialized at *`data/generated/output/<model-name>/model-serialized-<model-name>.pkl`*.
 5. Generate image-vs-label mapping for all the test images - `generate_images_labels.py test`.
 6. Test the model - `predict_from_file.py <model-name>`.
 7. To obtain Better Results, train the model using Convolutional Neural Network which can be done by running the cnn.py file after activating Tesorflow.
 8.Various other Supervised Models were trained which can be run by running- ` Final Python File.ipynb`
 9.Further SURF was implemented , which can be run by running - `SURF.ipynb`to get better results
 #### WorkFlow
 
 The complete WorkFlow can be explained in diagramatic form as follows
 <p align="center">
  <br>
  <img align="center" src="https://github.com/sanghaisubham/Indian-Sign-Language/blob/master/Workflow_Complete.PNG">
        <br>  
  </p>
  
 
 #### Accuracy without SURF
The accuracy of the various models tried without Feature Extraction is been shown as
follows:

<p align="center">
  <br>
  <img align="center" src="https://github.com/sanghaisubham/Indian-Sign-Language/blob/master/Accuracy_Without_SURF.PNG">
        <br>  
  </p>
  
  #### Accuracy using  SURF
The accuracy of the various models tried without Feature Extraction is been shown as
follows:

<p align="center">
  <br>
  <img align="center" src="https://github.com/sanghaisubham/Indian-Sign-Language/blob/master/Accuracy_Using_SURF.PNG">
        <br>  
  </p>
  
  #### Conclusion
  
In this project, attempts were made to achieve state of the art results for the Indian Sign Language like the ones that have been achieved for American Sign Language. The best accuracy was achieved by SVM after Feature Extraction using SURF which were invariant to to scaling,rotation .Codebooks were formed after feature extraction using SURF and K means .The accuracy reported here cannot be reported as a perfect representation of actual results because we are limited by data but can give us a direction as to which methods can be used when data is abundant.
  

