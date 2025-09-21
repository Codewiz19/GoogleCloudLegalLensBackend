# gcs_utils.py
import os
from typing import Optional
from google.oauth2 import service_account

def get_storage_client(creds_path: Optional[str] = None):
    """
    Return a google.cloud.storage.Client. If creds_path is provided, use that SA file.
    Otherwise use ADC (Application Default Credentials).
    Raise FileNotFoundError if the supplied creds_path doesn't exist.
    Raise RuntimeError with helpful instructions if client creation fails.
    """
    try:
        from google.cloud import storage
    except Exception as e:
        raise RuntimeError("Missing dependency 'google-cloud-storage'. Install with: pip install google-cloud-storage") from e

    if creds_path:
        creds_path = os.path.expanduser(creds_path)
        if not os.path.isfile(creds_path):
            raise FileNotFoundError(f"Service account JSON not found at: {creds_path}. "
                                    "Set GOOGLE_APPLICATION_CREDENTIALS correctly or remove it to use ADC (gcloud auth application-default login).")
        creds = service_account.Credentials.from_service_account_file(creds_path)
        try:
            client = storage.Client(project=os.environ.get("GCP_PROJECT"), credentials=creds)
            return client
        except Exception as e:
            raise RuntimeError(f"Failed to create storage client with provided credentials: {e}")
    else:
        # Try ADC
        try:
            client = storage.Client(project=os.environ.get("GCP_PROJECT"))
            return client
        except Exception as e:
            raise RuntimeError("Failed to create storage client via Application Default Credentials (ADC). "
                               "If running locally, run `gcloud auth application-default login` or set GOOGLE_APPLICATION_CREDENTIALS.") from e

def upload_file_to_gcs(local_path: str, bucket_name: str, dest_blob_name: str, creds_path: Optional[str] = None) -> str:
    """
    Upload local_path to bucket_name/dest_blob_name. Returns gs://... path on success.
    Raises informative exceptions on failure.
    """
    client = get_storage_client(creds_path)
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(dest_blob_name)
        blob.upload_from_filename(local_path)
        return f"gs://{bucket_name}/{dest_blob_name}"
    except Exception as e:
        raise RuntimeError(f"Failed to upload {local_path} to gs://{bucket_name}/{dest_blob_name}: {e}")
