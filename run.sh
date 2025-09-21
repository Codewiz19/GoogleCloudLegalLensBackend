#!/bin/bash
# Example run helper
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-/path/to/sa.json}"
export GCP_PROJECT="${GCP_PROJECT:-your-project-id}"
export GCP_LOCATION="${GCP_LOCATION:-us-east4}"
export STAGING_BUCKET="${STAGING_BUCKET:-gs://your-bucket}"
export SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_EMAIL:-your-sa@your-project.iam.gserviceaccount.com}"

uvicorn main:app --reload --host 0.0.0.0 --port 8000
