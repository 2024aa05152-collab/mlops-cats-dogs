#!/bin/bash
set -e

echo "Waiting for service to be ready..."
sleep 10

echo "Running health check..."

curl -f http://localhost:30007/health

echo "Running prediction test..."

curl -f -X POST http://localhost:30007/predict \
  -H "Content-Type: application/json" \
  -d '{"image":"test"}'

echo "Smoke tests passed!"
