function process_dpm_detection(model_path, images_dir, output_csv, image_list_file)
% PROCESS_DPM_DETECTION Process images using DPM model and save detection results
%
% Parameters:
%   model_path      - Path to model file (.mat file)
%   images_dir      - Directory containing images to process
%   output_csv      - Path to output CSV file
%   image_list_file - (Optional) Path to text file containing image filenames, if empty then scan directory

% Set up log file
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
log_file = ['dpm_detection_log_', timestamp, '.txt'];
diary(log_file);
diary on;

try
    % Display setup information
    fprintf('Starting DPM detection process...\n');
    fprintf('Model path: %s\n', model_path);
    fprintf('Images directory: %s\n', images_dir);
    fprintf('Output CSV: %s\n', output_csv);
    
    % Check if images directory exists
    if ~exist(images_dir, 'dir')
        error('Images directory does not exist: %s', images_dir);
    end
    
    % Try to load model
    fprintf('Loading model: %s\n', model_path);
    if ~exist(model_path, 'file')
        error('Model file does not exist: %s', model_path);
    end
    
    % Load model file
    model_data = load(model_path);
    
    % Determine correct model variable
    if isfield(model_data, 'model')
        model = model_data.model;
        fprintf('Using model field: model\n');
    else
        % Assume model is stored at top level
        model = model_data;
        fprintf('Using top-level model data\n');
    end
    
    % Get list of image files
    if ~isempty(image_list_file) && exist(image_list_file, 'file')
        % Read image paths from list file
        fprintf('Reading image paths from list file: %s\n', image_list_file);
        fid = fopen(image_list_file, 'r');
        image_files = textscan(fid, '%s');
        fclose(fid);
        image_files = image_files{1};
        
        % Process file paths
        for i = 1:length(image_files)
            % If relative path, add images_dir prefix
            if ~isfile(image_files{i}) && ~startsWith(image_files{i}, '/')
                image_files{i} = fullfile(images_dir, image_files{i});
            end
        end
    else
        % Scan directory for image files
        fprintf('Scanning directory for images: %s\n', images_dir);
        image_ext = {'.jpg', '.jpeg', '.png', '.bmp'};
        image_files = {};
        
        for i = 1:length(image_ext)
            files = dir(fullfile(images_dir, ['*', image_ext{i}]));
            for j = 1:length(files)
                image_files{end+1} = fullfile(images_dir, files(j).name);
            end
        end
    end
    
    num_images = length(image_files);
    fprintf('Found %d image files\n', num_images);
    
    if num_images == 0
        warning('No image files found');
        return;
    end
    
    % Create output file
    fid = fopen(output_csv, 'w');
    fprintf(fid, 'image_path,x1,y1,x2,y2,score\n');
    
    % Check if parallel computing toolbox is available
    try
        pool = gcp('nocreate');
        if isempty(pool)
            parpool('local');
            fprintf('Started parallel computing pool\n');
        else
            fprintf('Using existing parallel computing pool with %d workers\n', pool.NumWorkers);
        end
        use_parallel = true;
    catch ME
        fprintf('Could not start parallel computing pool: %s\n', ME.message);
        fprintf('Will use sequential processing\n');
        use_parallel = false;
    end
    
    % Process images
    if use_parallel
        fprintf('Starting parallel processing of %d images...\n', num_images);
        
        % Process images using parallel loop
        results = cell(num_images, 1);
        parfor i = 1:num_images
            results{i} = process_single_image(image_files{i}, model, i, num_images);
        end
        
        % Collect and save results
        for i = 1:num_images
            if ~isempty(results{i}) && ~isempty(results{i}.boxes)
                boxes = results{i}.boxes;
                
                for j = 1:size(boxes, 1)
                    fprintf(fid, '%s,%f,%f,%f,%f,%f\n', ...
                        image_files{i}, boxes(j,1), boxes(j,2), boxes(j,3), boxes(j,4), boxes(j,5));
                end
            end
        end
    else
        fprintf('Starting sequential processing of %d images...\n', num_images);
        
        % Process images sequentially
        for i = 1:num_images
            result = process_single_image(image_files{i}, model, i, num_images);
            
            if ~isempty(result) && ~isempty(result.boxes)
                boxes = result.boxes;
                
                for j = 1:size(boxes, 1)
                    fprintf(fid, '%s,%f,%f,%f,%f,%f\n', ...
                        image_files{i}, boxes(j,1), boxes(j,2), boxes(j,3), boxes(j,4), boxes(j,5));
                end
            end
        end
    end
    
    % Close output file
    fclose(fid);
    fprintf('Completed processing all images, results saved to: %s\n', output_csv);
    
catch ME
    % Catch and log errors
    fprintf('Error occurred:\n');
    fprintf('Error message: %s\n', ME.message);
    fprintf('Error details: %s\n', getReport(ME, 'extended'));
    
    % Ensure CSV file is closed
    if exist('fid', 'var') && fid ~= -1
        fclose(fid);
    end
    
    % Rethrow error for caller to handle
    rethrow(ME);
end

% Close log
diary off;
fprintf('Detection process completed, log saved to: %s\n', log_file);

end

function result = process_single_image(image_path, model, img_idx, total_imgs)
% Helper function to process a single image, containing only core steps

result = struct('boxes', []);

try
    fprintf('[%d/%d] Processing image: %s\n', img_idx, total_imgs, image_path);
    
    % Check if image file exists
    if ~exist(image_path, 'file')
        warning('Image file does not exist: %s', image_path);
        return;
    end
    
    % Read image
    im = imread(image_path);
    fprintf('  Successfully read image, size: [%d x %d x %d]\n', size(im, 1), size(im, 2), size(im, 3));
    
    % Perform detection - reference core steps from test function
    fprintf('  Performing detection...\n');
    
    % 1. Detect objects using imgdetect
    [ds, bs] = imgdetect(im, model, -1);
    
    % Check if any detections were found
    if isempty(ds)
        fprintf('  No objects detected\n');
        return;
    end
    
    % 2. Apply non-maximum suppression
    top = nms(ds, 0.5);
    if isempty(top)
        fprintf('  No detections remaining after NMS\n');
        return;
    end
    
    ds = ds(top, :);
    bs = bs(top, :);
    
    % 3. Apply bounding box prediction
    bbox = [];
    
    % Choose different post-processing based on model type
    if isfield(model, 'bboxpred')
        % Use bounding box prediction
        bbox = bboxpred_get(model.bboxpred, ds, reduceboxes(model, bs));
        bbox = clipboxes(im, bbox);
        top = nms(bbox, 0.5);
        bbox = bbox(top, :);
    else
        % No bounding box predictor, use reduceboxes directly
        bbox = reduceboxes(model, bs);
    end
    
    if isempty(bbox)
        fprintf('  Final bbox is empty\n');
        return;
    end
    
    % Return detection results
    result.boxes = bbox;
    fprintf('  Detection complete, found %d objects\n', size(bbox, 1));
    
catch ME
    % Log error but continue processing next image
    fprintf('  Error processing image: %s\n', ME.message);
end

end 