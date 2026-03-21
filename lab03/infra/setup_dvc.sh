#!/bin/bash

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

dvc remote add -d -f storage s3://pdiow
dvc remote modify storage endpointurl http://localhost:9000
dvc remote modify storage access_key_id "$MINIO_ROOT_USER"
dvc remote modify storage secret_access_key "$MINIO_ROOT_PASSWORD"