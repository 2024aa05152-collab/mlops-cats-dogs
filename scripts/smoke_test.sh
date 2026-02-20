#!/bin/bash
set -e

MINIKUBE_IP=$(minikube ip)
SERVICE_URL="http://$MINIKUBE_IP:30007"

IMAGE_PATH="data/processed/test/cats/83.jpg"

echo "Checking $SERVICE_URL"
echo "Using test image: $IMAGE_PATH"

echo "Waiting for rollout..."
sleep 15

echo "Encoding image..."
IMAGE_BASE64=$(base64 -w 0 $IMAGE_PATH)

echo "Running health check..."
curl -f $SERVICE_URL/health

echo "Running prediction test..."
curl -f -X POST $SERVICE_URL/predict \
  -H "Content-Type: application/json" \
  -d "{\"image\":\"$IMAGE_BASE64\"}"

echo "Smoke tests passed!"