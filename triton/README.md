# Triton model server

## Notes

```
podman run -it --rm --name=triton-server -p8000:8000 -p8002:8002 triton tritonserver --model-repository=./model_repository
```
```
...
...
...
+----------------------+---------+--------+
| Model                | Version | Status |
+----------------------+---------+--------+
| densenet_onnx        | 1       | READY  |
| fingerprint          | 1       | READY  |
| inception_graphdef   | 1       | READY  |
| simple               | 1       | READY  |
| simple_dyna_sequence | 1       | READY  |
| simple_identity      | 1       | READY  |
| simple_int8          | 1       | READY  |
| simple_sequence      | 1       | READY  |
| simple_string        | 1       | READY  |
+----------------------+---------+--------+
...
...
...
I1208 12:38:22.632432 1 grpc_server.cc:4820] Started GRPCInferenceService at 0.0.0.0:8001
I1208 12:38:22.632686 1 http_server.cc:3474] Started HTTPService at 0.0.0.0:8000
I1208 12:38:22.674873 1 http_server.cc:181] Started Metrics Service at 0.0.0.0:8002
```

```
model_repository
└── fingerprint
    └── 1
        └── model.savedmodel
            ├── saved_model.pb
```

Server config endpoints

```
curl 127.0.0.1:8000/v2
curl 127.0.0.1:8002/metrics
```

[Triton extentsions](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_model_repository.html#index)

```
curl -X POST ${HOST}/v2/repository/index | jq
curl localhost:8000/v2/models/fingerprint/config
```

Model info
```
curl localhost:8000/v2/models/fingerprint
```

[JSON formatter](https://jsonformatter.org/)

Inference using `curl`.
```
HOST=ec2-3-129-42-17.us-east-2.compute.amazonaws.com

curl -X POST -H "Content-Type: application/json" -d @request-fingerprint.json $HOST:8000/v2/models/fingerprint/infer | jq
```
```
{
  "model_name": "fingerprint",
  "model_version": "1",
  "outputs": [
    {
      "name": "dense_5",
      "datatype": "FP32",
      "shape": [
        1,
        1
      ],
      "data": [
        1
      ]
    }
  ]
}
```

```
{"model_name":"fingerprint","model_version":"1","outputs":[{"name":"dense_5","datatype":"FP32","shape":[1,1],"data":[1.0]}]}

```
