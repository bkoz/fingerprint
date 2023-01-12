# Triton model server

## Notes

### Openshift

#### Create the Triton model server application.
The Triton container is large and will take several minutes to build.

The following environment variables must be set to serve models
from a public s3 bucket.

- `APP_NAME=triton`
- `MODEL_REPOSITORY` (i.e. s3://mybucket/models/triton)
- `AWS_DEFAULT_REGION` (the region containing the s3 bucket)
```
oc new-app --name=${APP_NAME} --context-dir=/s2i-triton --strategy docker --env=AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} --env=MODEL_REPOSITORY=${MODEL_REPOSITORY} https://github.com/codekow/s2i-patch.git
```

To access an s3 bucket with authentication, additional environment variables must
be set.

```
oc new-app --name=${APP_NAME} --context-dir=/s2i-triton --strategy docker --env=AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} --env=AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY --env=AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} --env=MODEL_REPOSITORY=${MODEL_REPOSITORY} https://github.com/codekow/s2i-patch.git
```

After the pod gets deployed, expose the service and set the Openshift route
hostname to the `HOST` environment variable and test the server using the `curl` commands below.

Create an https route.
```
oc create route edge ${APP_NAME} --service=${APP_NAME} --port=8000
```

**Or** create an http route.
```
oc expose service ${APP_NAME}
```

#### Testing

Basic server test
```
export HOST=$(oc get route ${APP_NAME} --template={{.spec.host}})
curl https://${HOST}/v2 | python -m json.tool
```

Model metadata test
```
curl https://${HOST}/v2/models/fingerprint | python -m json.tool
```

Test using the python client program.
```
python client.py
```
```
INFO:root:Fingerprint Image = 103__F_Left_index_finger.BMP, Prediction = [0.0]
INFO:root:Fingerprint Image = 275__F_Left_index_finger.BMP, Prediction = [0.0]
INFO:root:Fingerprint Image = 232__M_Right_index_finger.BMP, Prediction = [1.0]
INFO:root:Fingerprint Image = 504__M_Right_index_finger.BMP, Prediction = [1.0]
```

#### Run with podman (need to update)
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
| fingerprint          | 1       | READY  |
+----------------------+---------+--------+
...
...
...
I1208 12:38:22.632432 1 grpc_server.cc:4820] Started GRPCInferenceService at 0.0.0.0:8001
I1208 12:38:22.632686 1 http_server.cc:3474] Started HTTPService at 0.0.0.0:8000
I1208 12:38:22.674873 1 http_server.cc:181] Started Metrics Service at 0.0.0.0:8002
```

Model repository directory structure.
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

[Triton extensions](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_model_repository.html#index)

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

curl -X POST -H "Content-Type: application/json" -d @scripts/request-fingerprint.json $HOST:8000/v2/models/fingerprint/infer | jq
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

### Notes from podman deployment

#### Using an S3 model repo with Triton

I had to use my personal AWS credentials as the Red Hat SAML creds
would not authenticate. Make sure each artifact in the s3 
bucket has public read permissions. When uploading a model folder to
an s3 bucket, choose permissions and grant public read access.

Set the following environment variables in the triton container:
```
AWS_DEFAULT_REGION
AWS_SECRET_ACCESS_KEY
AWS_ACCESS_KEY_ID
```

```
podman run -it --rm --name triton -p8000:8000 -p8002:8002 nvcr.io/nvidia/tritonserver:22.11-py3 /bin/bash

tritonserver --model-repository=s3://mybucket/models/triton --log-verbose=1

I1209 21:15:53.178161 218 filesystem.cc:2272] TRITON_CLOUD_CREDENTIAL_PATH environment variable is not set, reading from environment variables
I1209 21:15:53.178183 218 filesystem.cc:2350] Using credential    for path  s3://koz/models/triton
...
...
...
```
