# Fingerprint prediction

## Local client, model and data.

### Directory structure
```
.
├── data (sample fingerprint input images)
├── models (ML models)
├── predict.py
|── training (original ML training code)
|── triton (model serving)
```

### Prerequisites

- python v3.9
- pipenv

### Install
```
pipenv install
```

```
pipenv shell
```

### Run
```
python predict.py
```

or
```
pipenv run python predict.py
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

