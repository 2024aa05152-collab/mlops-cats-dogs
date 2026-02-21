#!/bin/bash
set -e

MINIKUBE_IP=$(minikube ip)
HOST="cats-dogs-model-group85.local"
BASE_URL="http://$MINIKUBE_IP"

IMAGE_PATH="data/processed/test/cats/83.jpg"

echo "Ingress host: $HOST"
echo "Minikube IP: $MINIKUBE_IP"
echo "Using test image: $IMAGE_PATH"

echo "Waiting for rollout..."
sleep 15

echo "Running health check..."
curl -f $BASE_URL/health \
  -H "Host: $HOST"

echo "Running prediction test..."
curl -f -X POST $BASE_URL/predict \
  -H "Host: $HOST" \
  -F "file=@$IMAGE_PATH"

echo "Running prediction with label test..."
curl -f -X POST $BASE_URL/predict_with_label \
  -H "Host: $HOST" \
  -F "file=@$IMAGE_PATH" \
  -F "true_label=cat"

echo "Smoke tests passed!"
