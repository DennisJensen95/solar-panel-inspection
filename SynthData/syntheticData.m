classdef syntheticData
    
    properties
    end
    
    methods (Static)
        
        function failures = count_failures(label_mem)
            %%%%%%%%%%%%%%%%%%%%
            %
            % Counting failures
            %
            %%%%%%%%%%%%%%%%%%%%
            
            % Types of failures
            failures = {'Finger Failure', 'Crack A', 'Crack B', 'Crack C'};
            
            finger_count = 0;
            crack_A = 0;
            crack_B = 0;
            crack_C = 0;
            
            for i = 1:length(label_mem)
                
                % Finger failure count
                if any(strcmp(label_mem{i}, failures{1}))
                    finger_count = finger_count + 1;
                end
                
                % Crack a count
                if any(strcmp(label_mem{i}, failures{2}))
                    crack_A = crack_A + 1;
                end
                
                % Crack B count
                if any(strcmp(label_mem{i}, failures{3}))
                    crack_B = crack_B + 1;
                end
                
                % Crack C count
                if any(strcmp(label_mem{i}, failures{4}))
                    crack_C = crack_C + 1;
                end
            end
            
            disp('How frequent a given failed is present per image')
            sprintf('Finger failure: %d. Crack A: %d. Crack B: %d. Crack C: %d', finger_count, crack_A, crack_B, crack_C)
            
            failures = [finger_count, crack_A, crack_B, crack_C];
        end
        
        function [newMask, newLabel] = edit_GTMask(mask, label, failure_type)
            if ~isempty(label) && length(label) < 2  
                idx = find(contains(label, failure_type));
                newLabel = label(idx);
                newMask = mask;
                newMask(newMask~=idx) = 0;
                newMask(newMask==idx) = 1;
            elseif ~isempty(label) && length(label) > 1
                idx = find(contains(label, failure_type));
                newLabel = label(idx(1));
                newMask = mask;
                newMask(newMask~=idx(1)) = 0;
                newMask(newMask==idx(1)) = 1;
            else
                mask = mask;
                label = label;
            end
                
        end
        
        function [mask_pin, image_pin] = pin_cushion_transform(image, mask)
            %%%%%%%%%%%%%%%%%%%%%%
            % Creating transformed images using PIN CUSHION METHOD
            %
            % image: the image you want to transform
            % mask: the mask that correspond to the image
            %
            %%%%%%%%%%%%%%%%%%%%%%
            
            % Pin cushion computation
            xmin = 0.01;
            xmax = 0.1;
            fill = 0.3;
            
            nrows = size(image, 1);
            ncols = size(image, 2);
            
            [xi,yi] = meshgrid(1:ncols,1:nrows);
            xt = xi - ncols/2;
            yt = yi - nrows/2;
            [theta,r] = cart2pol(xt,yt);
            rmax = max(r(:));
            
            b = xmin+rand(1)*(xmax-xmin);
            s = r - r.^3*(b/rmax.^2);
            
            [ut,vt] = pol2cart(theta,s);
            ui = ut + ncols/2;
            vi = vt + nrows/2;
            
            ifcn = @(c) [ui(:) vi(:)];
            tform = geometricTransform2d(ifcn);
            
            % Doing image transformation and saving
            mask_pin = imwarp(mask,tform,'FillValues',fill);
            image_pin = imwarp(image,tform,'FillValues',fill);
        end
        
        function [mask_rot, image_rot] = rotation_transform(image, mask)
            
            %%%%%%%%%%%%%%%%
            % Creating rotated duplicates
            %%%%%%%%%%%%%%%%
            
            Angles = [-270 -180 -90 0 90 180 270];
            
            % Choosing a rotation angle
            idx = randperm(length(Angles),1);
            rand_angle = Angles(idx);
            
            mask_rot = imrotate(mask, rand_angle);
            image_rot = imrotate(image, rand_angle);
        end
        
        function store_synth_data(image, GTMask, GTLabel, mask_filename, im_filename, saving_directory, start_name)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Saves corresponding image and .mat files into specified directory
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            Mask_newName = [mask_filename(1:(end-4)), start_name, '.mat'];
            save(fullfile(saving_directory{2}, Mask_newName), 'GTLabel', 'GTMask');
            
            Im_newName = [im_filename(1:(end-4)), start_name, '.png'];
            imwrite(image, fullfile(saving_directory{1}, Im_newName));
        end
    end
end