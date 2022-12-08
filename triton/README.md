# Triton model server

## Notes

```
podman run -it --rm --name=triton-server -p8000:8000 -p8002:8002 triton tritonserver --model-repository=./model_repository
```

```
model_repository
└── fingerprint
    └── 1
        └── model.savedmodel
            ├── assets
            ├── saved_model.pb
            └── variables
                ├── variables.data-00000-of-00001
                └── variables.index
```

Server config

```
curl 127.0.0.1:8000/v2
curl 127.0.0.1:8002/metrics
```

Model info
```
curl localhost:8000/v2/models/fingerprint
curl localhost:8000/v2/models/fingerprint/config
```

Inference (not working, example from a different model)
```
curl -X POST -H "Content-Type: application/json" -d '{"inputs": [ { "name": "predict", "shape": [1,64], "datatype": "FP32", "data": [[0.0, 0.0, 1.0, 11.0, 14.0, 15.0, 3.0, 0.0, 0.0, 1.0, 13.0, 16.0, 12.0, 16.0, 8.0, 0.0, 0.0, 8.0, 16.0, 4.0, 6.0, 16.0, 5.0, 0.0, 0.0, 5.0, 15.0, 11.0, 13.0, 14.0, 0.0, 0.0, 0.0, 0.0, 2.0, 12.0, 16.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 16.0, 16.0, 6.0, 0.0, 0.0, 0.0, 0.0, 16.0, 16.0, 16.0, 7.0, 0.0, 0.0, 0.0, 0.0, 11.0, 13.0, 12.0, 1.0, 0.0]] } ] }' 127.0.0.1:8000/v2/models/fingerprint/versions/1/infer
```
