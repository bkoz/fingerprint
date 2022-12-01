# Fingerprint prediction

## Setup

### Directory structure
```
.
├── data (sample fingerprint input images)
├── models (ML models)
├── predict.py
└── training (original ML training code)
```

### Prerequisites

- python v3.9
- pipenv

```
pipenv install
pipenv shell
```
```
python predict.py
```

Output
```
1/1 [==============================] - 0s 105ms/step
Prediction = [[1.]]
1/1 [==============================] - 0s 40ms/step
Prediction = [[1.]]
1/1 [==============================] - 0s 19ms/step
Prediction = [[0.]]
1/1 [==============================] - 0s 20ms/step
Prediction = [[0.]]
```

