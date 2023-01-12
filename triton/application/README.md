# Run on Openshift

```
APP_NAME=fingerprint
MODEL_SERVER_NAMESPACE=models
INFERENCE_HOST=$(oc get route triton -n ${MODEL_SERVER_NAMESPACE} --template={{.spec.host}})

oc new-app --name=${APP_NAME} --env=INFERENCE_HOST=${INFERENCE_HOST} --context-dir=/triton/application https://github.com/bkoz/fingerprint
```
