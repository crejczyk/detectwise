[![Build Status](https://travis-ci.org/crejczyk/detectwise.svg?branch=master)](https://travis-ci.org/crejczyk/detectwise)
[![GitHub license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/crejczyk/detectwise/blob/master/LICENSE)

# DetectWise

Real-Time Object Detection

## Run

```
mvn clean install exec:java
```

## YOLO(v1/v2) model to a Keras mode

#####  Download Darknet model cfg and weights from the [official YOLO website](http://pjreddie.com/darknet/yolo/).
```
git clone https://github.com/allanzelener/YAD2K.git
./yad2k.py yolo.cfg yolo.weights model_data/yolo.h5
run YoloKerasModelImport
```

## YOLO_v3 model to a Keras mode

#####  [Add YOLOv3 to model zoo](https://github.com/deeplearning4j/deeplearning4j/issues/4986)

## Built With

* [Maven](https://maven.apache.org/) - Dependency Management

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
