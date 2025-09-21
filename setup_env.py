#!/usr/bin/env python3
"""
Environment setup script for RAG FastAPI project.
This script helps set up the required environment variables.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create a .env file with required environment variables."""
    env_file = Path(".env")
    
    if env_file.exists():
        print("⚠️  .env file already exists. Backing up to .env.backup")
        env_file.rename(".env.backup")
    
    env_content = """# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
STAGING_BUCKET=gs://your-bucket-name
PROJECT_ID=your-gcp-project-id
LOCATION=us-central1

# Optional: Vertex AI Model Configuration
VERTEX_MODEL_NAME=gemini-1.5-pro
VERTEX_MAX_TOKENS=8192
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file with template values")
    print("📝 Please update the .env file with your actual Google Cloud credentials")

def check_requirements():
    """Check if required packages are installed."""
    try:
        import fastapi
        import uvicorn
        import google.cloud.aiplatform
        import google.cloud.storage
        import PyMuPDF
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("📦 Please run: pip install -r requirements.txt")
        return False

def main():
    print("🚀 Setting up RAG FastAPI Environment...")
    print()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    print()
    print("🎉 Setup complete!")
    print()
    print("Next steps:")
    print("1. Update the .env file with your Google Cloud credentials")
    print("2. Make sure your GCP project has Vertex AI API enabled")
    print("3. Create a Google Cloud Storage bucket for document storage")
    print("4. Run: python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    print()
    print("For more information, see the README.md file")

if __name__ == "__main__":
    main()
