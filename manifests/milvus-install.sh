#!/bin/bash
# =============================================================================
# Milvus Standalone Deployment on OpenShift
# =============================================================================

set -e

NAMESPACE="rag-pipeline"

echo "=== Step 1: Add Helm repo ==="
helm repo add zilliztech https://zilliztech.github.io/milvus-helm/
helm repo update

echo "=== Step 2: Grant SCC permissions (required for OpenShift) ==="
oc adm policy add-scc-to-user anyuid -z default -n $NAMESPACE
oc adm policy add-scc-to-user anyuid -z milvus-etcd -n $NAMESPACE
oc adm policy add-scc-to-user anyuid -z milvus-minio -n $NAMESPACE

echo "=== Step 3: Install Milvus standalone ==="
helm install milvus zilliztech/milvus \
  --namespace $NAMESPACE \
  --set cluster.enabled=false \
  --set standalone.resources.requests.cpu=500m \
  --set standalone.resources.requests.memory=2Gi \
  --set standalone.resources.limits.cpu=2 \
  --set standalone.resources.limits.memory=4Gi \
  --set streaming.enabled=false \
  --set woodpecker.enabled=false \
  --set pulsarv3.enabled=false \
  --set pulsar.enabled=false \
  --set etcd.replicaCount=1 \
  --set minio.mode=standalone \
  --set minio.resources.requests.memory=512Mi \
  --set etcd.resources.requests.memory=512Mi

echo "=== Step 4: Wait for pods ==="
echo "Milvus has a 90s readiness probe delay. Waiting..."
sleep 100

echo "=== Step 5: Verify ==="
oc get pods -n $NAMESPACE | grep milvus

echo ""
echo "Milvus endpoint: http://milvus.$NAMESPACE.svc.cluster.local:19530"
echo "Done!"
