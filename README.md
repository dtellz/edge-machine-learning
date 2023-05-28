# Machine Learning deployed at the edge

<p align='center'><img src='./03_deplyment/img/chip-brain.png' width='600' height='300'></p>

## Table of contents

1. [Tools to go deeper into ML knowledge](#tools-to-go-deeper-into-ml-knowledge)
2. [Toolkits for production AI application](#toolkits-for-production-ai-application)
3. [Introduction to TinyML](#introduction-to-tinyml)
4. [Applications of TinyML](#applications-of-tinyml)
5. [Deploying tiny ML in embedded systems](#deploying-tiny-ml-in-embedded-systems)
6. [Image - Visual Wake Words context](#image---visual-wake-words-context)
7. [Anomaly detection](#anomaly-detection)
8. [Metrics](#metrics)
9. [C++ Intro](./03_deployment/GUIDE_C%2B%2BIntro.md)
10. [Comunication protocols](./03_deployment/GUIDE_communication_protocols.md)
11. [Serial communication protocols](#serial-communication-protocols)
12. [Debugging microcontrollers](./03_deployment/GUIDE_debuggingMicrocontrollers.md)
13. [Emmbeded frameworks](#emmbeded-frameworks)
14. [Board sensors documentation](#board-sensors-documentation)
15. [Colabs](#colabs)
16. [Extra](#extra)

# Tools to go deeper into ML knowledge

[MIT Deep learning book - Recommended by OpenAI](https://www.deeplearningbook.org)

[Deep Reinforcement learning Tool - By OpenAi](https://spinningup.openai.com/en/latest/)

# Toolkits for production AI application
[Coral AI](https://coral.ai)

[Machinery Anomaly detection with vibration](https://www.mathworks.com/help/predmaint/ug/anomaly-detection-using-3-axis-vibration-data.html)

[Open speech recording for keyword spoting apps](https://tinyml.seas.harvard.edu/open_speech_recording)

[GAN: Generative Adversarial Networks AKA synthetic data generators](https://developers.google.com/machine-learning/gan)

[Real world project tutorials](https://coral.ai/examples/#project-tutorials)

[Open Source Speech datasets. KeyWordSpot, FullSentenceSegments, etc.](https://commonvoice.mozilla.org/en)

[Automatic hyperparameter optimization](https://cloud.google.com/automl)

# Introduction to TinyML

[how a Convolutional Neural Network can 'see' features in pictures](https://arxiv.org/pdf/1311.2901v3.pdf)

[how to visualize what a CNN is learning?? Display the features the network extracts from its filters](https://keras.io/examples/vision/visualizing_what_convnets_learn/)

# Applications of TinyML

[Saved models](https://www.tensorflow.org/guide/saved_model)

[Serving a model over HTTP](https://www.tensorflow.org/tfx/guide/serving)

[Tensorflow Lite](https://www.tensorflow.org/lite)

[Tensorflow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)

[Tensorflow JS](https://www.tensorflow.org/js)

[Running inference with saved models using tensorflow lite](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python)

[Post training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)

[Quantization aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training)

[Model conversion to TF Lite](https://www.tensorflow.org/lite/models/convert)

[Model checkpoin files exploration - hands on](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/checkpoint.ipynb)

[Model freezing in proto buffer format exploration - hands on](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/saved_model.ipynb)

# Deploying tiny ML in embedded systems

[C++ Official documentation](https://learn.microsoft.com/en-us/cpp/cpp/?view=msvc-170)

[C++ Simplified doc & tutorials](https://cplusplus.com/doc/tutorial/)

[Google's C++ Style guide](https://google.github.io/styleguide/cppguide.html)

[C++ variables/functions/structures to control arduino](https://www.arduino.cc/reference/en/)

[Full list of Arduino libraries](https://www.arduino.cc/reference/en/libraries/)

## Serial communication protocols

[Universal asynchronous receive transmit (UART)](https://learn.sparkfun.com/tutorials/serial-communication/all)

[I2C protocol](https://learn.sparkfun.com/tutorials/i2c)

[Serial peripheral interface (SPI)](https://learn.sparkfun.com/tutorials/serial-peripheral-interface-spi)

## Image - Visual Wake Words context

[Depthwise Separable Convolutions - // Logic behind mobilenet](https://learning.edx.org/course/course-v1:HarvardX+TinyML2+1T2022/block-v1:HarvardX+TinyML2+1T2022+type@sequential+block@7f21dd62ca5344a98653f0c8a4c877c6/block-v1:HarvardX+TinyML2+1T2022+type@vertical+block@5bc2a4040ae245f294825c8a404f42be)

[MobileNet Paper](https://arxiv.org/pdf/1704.04861.pdf)

[Image preprocess and management in Tensorflow](https://www.tensorflow.org/tutorials/load_data/images)

[buffered prefecth technique to improve data performance with TensorFlow API](https://www.tensorflow.org/guide/data_performance)

[Overfit and underfit](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)

[Data augmentation - improve model quality and diversity applying zoom and rotation to images](https://www.tensorflow.org/tutorials/images/data_augmentation)

[Pixel re-escaling layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling)

[Functional layers API](https://www.tensorflow.org/guide/keras/functional)

# Anomaly detection

[MIMII Dataset paper: sound dataset for malfunctioning industrial machines](https://arxiv.org/pdf/1909.09347.pdf)

[Link to MIMII Dataset](https://zenodo.org/record/3384388#.ZDxLBy8w0eY)

[Variational autoencoders](https://www.tensorflow.org/tutorials/generative/cvae)

[Electro CardioGram dataset](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000)

[Anomaly detection with k-means clustering](http://amid.fish/anomaly-detection-with-k-means-clustering)

[Building autoencoders in keras](https://blog.keras.io/building-autoencoders-in-keras.html)

# Metrics

[Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)

[Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision)

[Precision](https://en.wikipedia.org/wiki/Precision_and_recall)

[false positive rate](https://en.wikipedia.org/wiki/False_positive_rate)

[f1 score](https://en.wikipedia.org/wiki/F-score)

[cross validation](https://machinelearningmastery.com/k-fold-cross-validation/)

# Colabs

[Mask detection with transfer learning / mobileNet V1](https://colab.research.google.com/github/tinyMLx/colabs/blob/master/3-7-11-Assignment.ipynb)

## Extra
[Assigments, projects and more info from Harvard's onsite course](https://sites.google.com/g.harvard.edu/tinyml/home)

[Harvard's maching learning community](https://discuss.tinyml.seas.harvard.edu/latest)

[Harvard's TinyML full course set](https://tinyml.seas.harvard.edu/courses/)

## Board sensors documentation

[Microphone: MP34DT05-A](https://www.st.com/resource/en/datasheet/mp34dt05-a.pdf)

[Acceleromenter/gyroscope/magnetometer: LSM9DS1](https://www.st.com/resource/en/datasheet/lsm9ds1.pdf)

[Camera: OV7675](https://www.uctronics.com/download/Image_Sensor/OV7675_DS.pdf)

## Emmbeded frameworks

[Arduino](https://www.arduino.cc/)

[PlatformIO](https://platformio.org/)

[ARM cortex - CMSIS](https://developer.arm.com/tools-and-software/embedded/cmsis)

[freeRTOS](https://www.freertos.org/)

[mbed](https://os.mbed.com/)

[STM32CubeMX](https://www.st.com/en/development-tools/stm32cubemx.html)
