import subprocess
import sys
import os

# Global list to store failed commands
failed_commands = []

def run_command(command):
    """
    Runs a shell command and prints its output.
    Records failures but does not exit.
    """
    print(f"Executing: {command}")
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='')
        process.wait() # Wait for the subprocess to complete
        if process.returncode != 0:
            print(f"Warning: Command '{command}' failed with exit code {process.returncode}")
            failed_commands.append(command)
            return False # Indicate failure
        return True # Indicate success
    except Exception as e:
        print(f"Warning: An exception occurred while trying to run command '{command}': {e}")
        failed_commands.append(command)
        return False # Indicate failure

def install_dependencies():
    """
    Installs all the required Python dependencies, continuing on failure.
    """
    print("--- Starting dependency installation ---")

    # Install numpy first to satisfy build dependencies for other packages
    print("\n--- Installing numpy first (critical build dependency) ---")
    run_command("pip install -U numpy")

    # Core Python dependencies (excluding numpy as it's already installed)
    print("\n--- Installing core Python dependencies (remaining) ---")
    run_command("pip install -U simplejson psutil tqdm PyYAML iopath opencv-python tensorboard moviepy matplotlib pandas scikit-learn av plotly imutils")

    # Install fvcore and fairscale from source
    print("\n--- Installing fairscale from source ---")
    run_command("pip install 'git+https://github.com/facebookresearch/fairscale'")

    print("\n--- Installing torch, torchvision, cython ---")
    run_command("pip install -U torch torchvision cython")

    print("\n--- Installing fvcore from source ---")
    run_command("pip install -U 'git+https://github.com/facebookresearch/fvcore.git'")

    # Now install cocoapi, numpy is already available
    print("\n--- Installing cocoapi from source ---")
    run_command("pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'")

    # Uninstall existing torch and install specific version
    print("\n--- Uninstalling existing torch and installing specific CUDA-enabled version ---")
    run_command("pip uninstall -y torch")
    run_command("pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 -f https://download.pytorch.org/whl/torch_stable.html")

    # Install detectron2
    print("\n--- Installing detectron2 ---")
    run_command("pip install git+https://github.com/facebookresearch/detectron2@7c2c8fb")

    # Install pytorchvideo
    print("\n--- Installing pytorchvideo ---")
    run_command("pip install 'git+https://github.com/facebookresearch/pytorchvideo.git'")

    # Set PYTHONPATH (this is a shell export, so for a Python script, it needs to be handled differently or assume
    # the user will set it in their environment before running their main application if needed for slowfast)
    # For a persistent effect within the script's execution context, we can modify os.environ
    print("\n--- Setting PYTHONPATH for slowfast (if applicable) ---")
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    # Get the directory of the current script, then assume slowfast is a sibling directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    slowfast_path = os.path.join(script_dir, 'slowfast')
    # Make sure to expanduser to handle '~' if it were ever used (though less common with os.getcwd())
    slowfast_path = os.path.abspath(slowfast_path)

    # Only add if it's not already in the PYTHONPATH
    if slowfast_path not in current_pythonpath.split(os.pathsep):
        os.environ['PYTHONPATH'] = f"{slowfast_path}{os.pathsep}{current_pythonpath}" if current_pythonpath else slowfast_path
        print(f"Updated PYTHONPATH: {os.environ['PYTHONPATH']}")
    else:
        print("slowfast path already in PYTHONPATH.")


    # Install specific ultralytics and Pillow
    print("\n--- Installing ultralytics and Pillow ---")
    run_command("pip install -U ultralytics")
    run_command("pip install Pillow==9.5.0")

    print("\n--- Dependency installation attempt complete ---")

    # Summary of failures
    print("\n--- Installation Summary ---")
    if failed_commands:
        print("\nThe following commands failed during execution:")
        for cmd in failed_commands:
            print(f"- {cmd}")
        print("\nPlease review the output above for specific error messages for these commands.")
    else:
        print("\nAll commands completed successfully (though individual package installations might still have warnings).")

    # Verify CUDA availability
    print("\n--- Verifying PyTorch CUDA availability ---")
    try:
        import torch
        print(f"Torch can access CUDA : {torch.cuda.is_available()}")
    except ImportError:
        print("Error: torch could not be imported. CUDA availability check skipped.")
    except Exception as e:
        print(f"An unexpected error occurred during CUDA check: {e}")


if __name__ == "__main__":
    install_dependencies()
