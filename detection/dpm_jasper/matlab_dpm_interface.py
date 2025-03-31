import os
import sys
import argparse
import matlab.engine


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Interface to MATLAB DPM detection')
    
    parser.add_argument('--images-dir', type=str, default='cow_dataset/images/',
                      help='Directory containing images to process (default: cow_dataset/images/)')
    
    parser.add_argument('--output-csv', type=str, default='cow_detection_results.csv',
                      help='Output CSV file path (default: cow_detection_results.csv)')
    
    parser.add_argument('--model-path', type=str, default=None,
                      help='Path to the model file (default: will search for cow_final.mat)')
    
    parser.add_argument('--matlab-path', type=str, default=None,
                      help='Path to the MATLAB code directory containing DPM model')
    
    parser.add_argument('--image-list', type=str, default=None,
                      help='Path to a text file containing image filenames to process')
    
    return parser.parse_args()


def find_model_file(base_dirs):
    """Find the model file by searching in common locations"""
    model_names = ['cow_final.mat', 'cow_model.mat']
    model_subdirs = ['', 'dpm/2012', 'dpm', 'voc-dpm/2012', 'voc-dpm']
    
    for base_dir in base_dirs:
        for subdir in model_subdirs:
            for model_name in model_names:
                path = os.path.join(base_dir, subdir, model_name)
                if os.path.isfile(path):
                    return path
    
    return None


def main():
    """Main function to interface with MATLAB DPM detection"""
    # Parse command line arguments
    args = parse_args()
    
    # Prepare paths
    images_dir = os.path.abspath(args.images_dir)
    output_csv = args.output_csv
    
    # Check if images directory exists
    if not os.path.exists(images_dir):
        print(f"Error: Images directory '{images_dir}' does not exist")
        return 1
    
    # Find model file
    model_path = args.model_path
    if model_path is None:
        print(f"No model path specified, searching for model file...")
        search_dirs = [os.getcwd()]
        model_path = find_model_file(search_dirs)
        
        if model_path is None:
            print("Error: Could not find model file automatically. Please specify with --model-path")
            return 1
    
    model_path = os.path.abspath(model_path)
    print(f"Using model file: {model_path}")
    
    # Check if model file exists
    if not os.path.isfile(model_path):
        print(f"Error: Model file '{model_path}' does not exist")
        return 1
    
    # Check image list
    image_list = args.image_list
    if image_list is not None:
        image_list = os.path.abspath(image_list)
        if not os.path.isfile(image_list):
            print(f"Warning: Image list file '{image_list}' does not exist, will scan directory instead")
            image_list = None
    
    # Start MATLAB engine
    print("Starting MATLAB engine...")
    try:
        eng = matlab.engine.start_matlab()
        print("MATLAB engine started successfully")
    except Exception as e:
        print(f"Failed to start MATLAB engine: {str(e)}")
        return 1
    
    try:
        # Set MATLAB working directory
        matlab_path = args.matlab_path
        
        if matlab_path is None:
            # Try to find the directory containing the model file
            matlab_path = os.path.dirname(model_path)
            
            # If that doesn't work, try common directories
            if not os.path.exists(os.path.join(matlab_path, "startup.m")):
                possible_paths = [
                    os.path.join(os.getcwd(), "voc-dpm"),
                    os.path.join(os.getcwd(), "dpm"),
                    os.getcwd()
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        matlab_path = path
                        print(f"Using detected MATLAB path: {matlab_path}")
                        break
        
        if matlab_path:
            print(f"Setting MATLAB working directory to: {matlab_path}")
            eng.cd(matlab_path, nargout=0)
        
        # Get current MATLAB directory for debugging
        current_dir = eng.pwd(nargout=1)
        print(f"MATLAB current directory: {current_dir}")
        
        # Run startup.m to ensure all paths are correctly set
        try:
            print("Running startup.m...")
            eng.eval("startup", nargout=0)
            print("MATLAB paths initialized successfully")
        except Exception as e:
            print(f"Warning: Could not run startup.m: {str(e)}")
            print("Continuing anyway, but detection might fail if paths are not set correctly")
        
        # Check if process_dpm_detection.m exists in the current directory
        script_file = os.path.join(current_dir, "process_dpm_detection.m")
        if not os.path.isfile(script_file):
            # If not, check if it's in the current Python script directory
            script_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "process_dpm_detection.m")
            if os.path.isfile(script_file):
                # Copy or add to path
                print(f"Found process_dpm_detection.m at {script_file}")
                eng.addpath(os.path.dirname(script_file), nargout=0)
            else:
                print("Error: Could not find process_dpm_detection.m")
                return 1
        
        # Run MATLAB detection function
        print(f"Running DPM detection on images in {images_dir}")
        print(f"Output will be saved to {output_csv}")
        
        # Convert paths to format MATLAB can handle
        model_path_ml = model_path.replace('\\', '/')
        images_dir_ml = images_dir.replace('\\', '/')
        output_csv_ml = output_csv.replace('\\', '/')
        image_list_ml = image_list.replace('\\', '/') if image_list else ""
        
        # Call MATLAB function
        if image_list:
            eng.process_dpm_detection(model_path_ml, images_dir_ml, output_csv_ml, image_list_ml, nargout=0)
        else:
            eng.process_dpm_detection(model_path_ml, images_dir_ml, output_csv_ml, "", nargout=0)
        
        print("MATLAB processing completed successfully")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Close MATLAB engine
        print("Closing MATLAB engine...")
        eng.quit()
    
    print("Processing completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 