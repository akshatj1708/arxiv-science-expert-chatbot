import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)

def create_virtualenv():
    """Create a virtual environment if it doesn't exist."""
    venv_dir = Path("venv")
    if not venv_dir.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    else:
        print("Virtual environment already exists.")

def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    pip_cmd = ["venv/bin/pip", "install", "-r", "requirements.txt"]
    if sys.platform == "win32":
        pip_cmd[0] = "venv\\Scripts\\pip"
    
    try:
        subprocess.run(pip_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

def setup_environment():
    """Set up the environment variables."""
    env_file = Path(".env")
    if not env_file.exists():
        print("Creating .env file from example...")
        try:
            with open(".env.example", "r") as src, open(".env", "w") as dst:
                dst.write(src.read())
        except Exception as e:
            print(f"Error creating .env file: {e}")
    else:
        print(".env file already exists.")

def main():
    print("Setting up arXiv Research Assistant...")
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    create_virtualenv()
    
    # Install requirements
    install_requirements()
    
    # Set up environment
    setup_environment()
    
    print("\nSetup complete!\n")
    print("To start the application, run:")
    if sys.platform == "win32":
        print("  .\\venv\\Scripts\\activate")
        print("  streamlit run app.py")
    else:
        print("  source venv/bin/activate")
        print("  streamlit run app.py")
    print("\nThen open your browser to http://localhost:8501")

if __name__ == "__main__":
    main()
