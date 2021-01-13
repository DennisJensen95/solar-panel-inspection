%% Init
close all
clc
clear

%% Synth data filter
% Select which kind of failure you want to upload
finger_fail_gen = false;
crack_A_gen = true;
crack_B_gen = true;
crack_C_gen = true;

% Show images
debug = false;

% Select pin cushion
pin_cushion = true; % Uses pin cushion transformation

failures = {'Finger Failure', 'Crack A', 'Crack B', 'Crack C'};

% Saving directory
outDirCells = fullfile('../data/synth_data/CellsCorr');
outDirMask = fullfile('../data/synth_data/MaskGT');
saving_directories = {outDirCells, outDirMask};

% Amount of data desired total (with existing data)
data_des = 400;
%% Load available data

addpath("../data/Serie1_cellsAndGT/CellsCorr");
addpath("../data/Serie1_cellsAndGT/MaskGT");

synth = syntheticData;

if ~exist('../data/synth', 'dir') && ~exist('../data/synth_data', 'dir') &&...
        ~exist('../data/MaskGT', 'dir') && ~exist('../data/combined_data', 'dir')...
        && ~exist('../data/combined_data/CellsCorr', 'dir') && ~exist('../data/combined_data/MaskGT', 'dir')
    mkdir ../data/synth_data
    mkdir ../data/synth_data/CellsCorr
    mkdir ../data/synth_data/MaskGT
    mkdir ../data/combined_data
    mkdir ../data/combined_data/CellsCorr
    mkdir ../data/combined_data/MaskGT
else
    sprintf('The synthetic data directories already exists!')
end

[~,~,T] = xlsread("available_files.csv");
cells = regexp(T, ',', 'split');

for i = 1:(length(cells))
    available_im(i) = cellstr(cells{i}{1});
    available_mask(i) = cellstr(cells{i}{2});
end

available_im(1) = [];
available_mask(1) = [];


for i = 1:length(available_im)
    Images(i) = dir(fullfile(available_im{i}));
    Masks(i) = dir(fullfile(available_mask{i}));
end

for k = 1:length(Images)
    % Masks
    Mask_filename = Masks(k).name;
    info = load(Mask_filename);
    mask = info.GTMask;
    
    % Storing all labels
    label = info.GTLabel;
    label_memory_check{k} = label;
end

failuresN = synth.count_failures(label_memory_check);

%% Creating new data

pin_name = '';


for k = 1:length(Images)
    
    % Masks
    Mask_filename = Masks(k).name;
    info = load(Mask_filename);
    mask = info.GTMask;
    
    % Storing all labels
    label = info.GTLabel;
    label_memory{k} = label;
    
    % Loading images
    Im_filename = Images(k).name;
    image = imread(available_im{k});
    
    
    if finger_fail_gen && any(strcmp(label_memory{k},failures{1}))
        
        [mask_cp, label_rdy] = synth.edit_GTMask(mask, label, failures{1});
        dataN = data_des - failuresN(1);
        
        if ~dataN < 0
            for i = 1:(round(dataN/failuresN(1)))
                if pin_cushion
                    [mask_cp, image_cp] = synth.pin_cushion_transform(image, mask_cp);
                    pin_name = '_pin';
                end
                [mask_rdy, image_rdy] = synth.rotation_transform(image_cp, mask_cp);
                rot_name = '_rot';
                synth.store_synth_data(image_rdy, mask_rdy, label_rdy, Mask_filename,...
                    Im_filename, saving_directories, strcat('_iter_',num2str(i),'_','finger',pin_name,rot_name))
                if debug
                    synth.showImage(mask_rdy, image_rdy)
                    pause;
                    disp('skipped');
                end
            end
        else
            continue
        end
    end
    
    if crack_A_gen && any(strcmp(label_memory{k},failures{2}))
        
        [mask_cp, label_rdy] = synth.edit_GTMask(mask, label, failures{2});
        dataN = data_des - failuresN(2);
        
        for i = 1:(round(dataN/failuresN(2)))
            if pin_cushion
                [mask_cp, image_cp] = synth.pin_cushion_transform(image, mask_cp);
                pin_name = '_pin';
            end
            [mask_rdy, image_rdy] = synth.rotation_transform(image_cp, mask_cp);
            rot_name = '_rot';
            synth.store_synth_data(image_rdy, mask_rdy, label_rdy, Mask_filename,...
                Im_filename, saving_directories, strcat('_iter_',num2str(i),'_','cA',pin_name,rot_name))
            if debug
                synth.showImage(mask_rdy, image_rdy)
                pause;
                disp('skipped');
            end
        end
    end
    
    if crack_B_gen && any(strcmp(label_memory{k},failures{3}))
        
        [mask_cp, label_rdy] = synth.edit_GTMask(mask, label, failures{3});
        dataN = data_des - failuresN(3);
        
        for i = 1:(round(dataN/failuresN(3)))
            if pin_cushion
                [mask_cp, image_cp] = synth.pin_cushion_transform(image, mask_cp);
                pin_name = '_pin';
            end
            [mask_rdy, image_rdy] = synth.rotation_transform(image_cp, mask_cp);
            rot_name = '_rot';
            synth.store_synth_data(image_rdy, mask_rdy, label_rdy, Mask_filename,...
                Im_filename, saving_directories, strcat('_iter_',num2str(i),'_','cB',pin_name,rot_name))
            if debug
                synth.showImage(mask_rdy, image_rdy)
                disp(Im_filename)
                pause;
                disp('skipped');
            end
        end
    end
    
    if crack_C_gen && any(strcmp(label_memory{k},failures{4}))
        
        % Editing the mask to contain only 1 label, this case, Crack C
        [mask_cp, label_rdy] = synth.edit_GTMask(mask, label, failures{4});
        dataN = data_des - failuresN(4);
        
        for i = 1:(round(dataN/failuresN(4)))
            
            if pin_cushion
                % Pin cushion transformation
                [mask_cp, image_cp] = synth.pin_cushion_transform(image, mask_cp);
                
                % For naming the output file
                pin_name = '_pin';
            end
            % Rotating the image
            [mask_rdy, image_rdy] = synth.rotation_transform(image_cp, mask_cp);
            
            % For naming the output file
            rot_name = '_rot';
            
            % Storing image and (mask,label)
            synth.store_synth_data(image_rdy, mask_rdy, label_rdy, Mask_filename,...
                Im_filename, saving_directories, strcat('_iter_',num2str(i),'_','cC',pin_name,rot_name))
            
            if debug
                synth.showImage(mask_rdy, image_rdy)
                pause;
                disp('skipped');
            end
        end
    end
    
    
end
%% Counting the failures

% synth.count_failures(label_memory)

  
            