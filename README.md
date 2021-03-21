# face mask detector

## Overview
The system for recognizing the presence 
of a medical mask in a photo of a person

## Local run
* Check you have Python 3.7.2 installed by typing on commandline `python --version`
* Install environment
```
pip install virtualenv
virtualenv venv
CALL venv/Scripts/activate (on Windows)
source venv/bin/activate (on Linux)
pip install -r requirements.txt
cd webapp/
```
* Run server `python manage.py runserver`
* Point your web browser to http://localhost:5000/

## Neural network module
* Dataset structure in project hierarchy:
```
face_mask_detector
├───neural_network
│   └───dataset
│       ├───images-of-people-not-wearing-face-masks
│       └───images-of-people-wearing-face-masks
```
* You can retrain prediction model. Please use:
```
cd neural_network/
python train_face_mask_detector.py
```
* For check neural network, next script can be used:
```
cd neural_network/
python test.py
```

