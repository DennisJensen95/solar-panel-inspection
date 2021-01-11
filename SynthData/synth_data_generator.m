%% Init
close all
clc
clear

%% Synth data filter
% Select which kind of failure you want to upload
finger_fail_gen = false;
crack_A_gen = false;
crack_B_gen = false;
crack_C_gen = true;
all_failures = false;

% How many times to repeat data generation
repeats = 5;

% Select pin cushion
pin_cushion = false; % Uses pin cushion transformation 

% Make a directory with synth data
make_dir = false;

failures = {'Finger Failure', 'Crack A', 'Crack B', 'Crack C'};

% Saving directory
outDirCells = fullfile('../data/synth_data/CellsCorr');
outDirMask = fullfile('../data/synth_data/MaskGT');
saving_directories = {outDirCells, outDirMask};

% Amount of data generated
data_des = 300;
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

%% Creating new data

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
pin_name = '';


for k = 1:length(Images)
    
    % Masks
    Mask_filename = Masks(k).name;
    info = load(Mask_filename);
    mask = info.GTMask;
    
    % Storing all labels
    label = info.GTLabel;
    label_memory{k} = label;
    
    % For pin cushion image transformation
    Im_filename = Images(k).name;
    image = imread(Im_filename);
    
    if finger_fail_gen && any(strcmp(label_memory{k},failures{1}))
        [mask, label] = synth.edit_GTMask(mask, label, failures{1});
        dataN = data_des - failuresN(1);
        if ~dataN < 0 
            for i = 1:(round(dataN/failuresN(1)))
                if pin_cushion
                    [mask, image] = synth.pin_cushion_transform(image, mask);
                    pin_name = '_pin';
                end
                [mask, image] = synth.rotation_transform(image, mask);
                rot_name = '_rot';
                synth.store_synth_data(image, mask, label, Mask_filename,...
                    Im_filename, saving_directories, strcat('_iter_',num2str(i),'_','finger',pin_name,rot_name))
            end
        else
            continue
        end
    end
    
    if crack_A_gen && any(strcmp(label_memory{k},failures{2}))
        [mask, label] = synth.edit_GTMask(mask, label, failures{2});
        dataN = data_des - failuresN(2);
        for i = 1:(round(dataN/failuresN(2)))
            if pin_cushion
                [mask, image] = synth.pin_cushion_transform(image, mask);
                pin_name = '_pin';
            end
            [mask, image] = synth.rotation_transform(image, mask);
            rot_name = '_rot';
            synth.store_synth_data(image, mask, label, Mask_filename,...
                Im_filename, saving_directories, strcat('_iter_',num2str(i),'_','cA',pin_name,rot_name))
        end
    end
    
    if crack_B_gen && any(strcmp(label_memory{k},failures{3}))
        [mask, label] = synth.edit_GTMask(mask, label, failures{3});
        dataN = data_des - failuresN(3);
        for i = 1:(round(dataN/failuresN(3)))
            if pin_cushion
                [mask, image] = synth.pin_cushion_transform(image, mask);
                pin_name = '_pin';
            end
            [mask, image] = synth.rotation_transform(image, mask);
            rot_name = '_rot';
            synth.store_synth_data(image, mask, label, Mask_filename,...
                Im_filename, saving_directories, strcat('_iter_',num2str(i),'_','cB',pin_name,rot_name))
        end
    end
    
    if crack_C_gen && any(strcmp(label_memory{k},failures{4}))
        [mask, label] = synth.edit_GTMask(mask, label, failures{4});
        dataN = data_des - failuresN(4);
        for i = 1:(round(dataN/failuresN(4)))
            if pin_cushion
                [mask, image] = synth.pin_cushion_transform(image, mask);
                pin_name = '_pin';
            end
            [mask, image] = synth.rotation_transform(image, mask);
            rot_name = '_rot';
            synth.store_synth_data(image, mask, label, Mask_filename,...
                Im_filename, saving_directories, strcat('_iter_',num2str(i),'_','cC',pin_name,rot_name))
        end
    end
    

end
%% Counting the failures

synth.count_failures(label_memory)

  
            