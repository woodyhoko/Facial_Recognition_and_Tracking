function [ppp] = track_r(ppp,firsttime)
global objh;
global img;
if (firsttime==1)
    firsttime=0;
    [thisname,videoPlayer, base_target_sz, block_inds, cam, CG_opts, cos_window, currentScaleFactor, debugshow, distance_matrix, feature_dim, feature_extract_info, feature_info, feature_params, feature_sz, feature_sz_cell, features, filter_sz, filter_sz_cell, frames_since_last_train, global_fparams, gram_matrix, img_sample_sz, img_support_sz, init_CG_opts, init_target_sz, interp1_fs, interp2_fs, is_color_image, k, k1, kx, ky, latest_ind, max_scale_factor, min_scale_factor, nScales, num_feature_blocks, num_training_samples, output_sz, pad_sz, params, pos, prior_weights, reg_energy, reg_filter, reg_window_edge, res_norms, residuals_pcg, sample_dim, sample_weights, samplesf, scale_filter, scale_step, scaleFactors, scores_fs_feat, search_area, seq, sig_y, target_sz, yf, yf_x, yf_y,to]=objh{ppp}{1:end};
else
    [thisname,videoPlayer, base_target_sz, block_inds, cam, CG_opts, CG_state, cos_window, currentScaleFactor, debugshow, distance_matrix, feature_dim, feature_extract_info, feature_info, feature_params, feature_sz, feature_sz_cell, features, filter_sz, filter_sz_cell, frames_since_last_train, global_fparams, gram_matrix, hf, hf_full, img_sample_sz, img_support_sz, init_CG_opts, init_target_sz, interp1_fs, interp2_fs, is_color_image, k, k1, kx, ky, latest_ind, max_scale_factor, merged_sample, merged_sample_id, min_scale_factor, new_sample, new_sample_energy, new_sample_id, new_train_sample_norm, nScales, num_feature_blocks, num_training_samples, output_sz, pad_sz, params, pos, ppp, prior_weights, proj_energy, projection_matrix, rect_position_vis, reg_energy, reg_filter, reg_window_edge, res_norms, residuals_pcg, sample_dim, sample_energy, sample_pos, sample_scale, sample_weights, samplesf, scale_filter, scale_step, scaleFactors, scores_fs_feat, search_area, seq, shift_samp, sig_y, target_sz, tracking_result, train_tracker, xl, xlf, xlf_proj, xlf_proj_perm, yf, yf_x, yf_y,to]=objh{ppp}{1:end};
end
    % Read image

    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        im=img;
        if isempty(im)
            return;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        [seq, im] = get_sequence_frame(seq);
        im = img;
        seq.frame = 1;
    end

    %tic();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Do not estimate translation and scaling on the first frame, since we 
    % just want to initialize the tracker there
    if seq.frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            sample_pos = round(pos);
            det_sample_pos = sample_pos;
            sample_scale = currentScaleFactor*scaleFactors;
            xt = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_extract_info);
                        
            % Project sample
            xt_proj = project_sample(xt, projection_matrix);
            
            % Do windowing of features
            xt_proj = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt_proj, cos_window, 'uniformoutput', false);
            
            % Compute the fourier series
            xtf_proj = cellfun(@cfft2, xt_proj, 'uniformoutput', false);
            
            % Interpolate features to the continuous domain
            xtf_proj = interpolate_dft(xtf_proj, interp1_fs, interp2_fs);
            
            % Compute convolution for each feature block in the Fourier domain
            % and the sum over all blocks.
            scores_fs_feat{k1} = sum(bsxfun(@times, hf_full{k1}, xtf_proj{k1}), 3);
            scores_fs_sum = scores_fs_feat{k1};
            
            
            % Also sum over all feature blocks.
            % Gives the fourier coefficients of the convolution response.
            scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);
            
            % Optimize the continuous score function with Newton's method.
            [trans_row, trans_col, scale_ind] = optimize_scores(scores_fs, params.newton_iterations);
            
            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.
            translation_vec = [trans_row, trans_col] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(scale_ind);
            scale_change_factor = scaleFactors(scale_ind);
            
            % update position
            old_pos = pos;
            pos = sample_pos + translation_vec;
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
            
            % Do scale tracking with the scale filter
            if nScales > 0 && params.use_scale_filter
                scale_change_factor = scale_filter_track(im, pos, base_target_sz, currentScaleFactor, scale_filter, params);
            end 
            
            % Update the scale
            currentScaleFactor = currentScaleFactor * scale_change_factor;
            
            % Adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            
            iter = iter + 1;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    % Extract sample and init projection matrix
    if seq.frame == 1
        % Extract image region for training sample
        sample_pos = round(pos);
        sample_scale = currentScaleFactor;
        xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
        
        % Do windowing of features
        xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
        
        % Compute the fourier series
        xlf = cellfun(@cfft2, xlw, 'uniformoutput', false);
        
        % Interpolate features to the continuous domain
        xlf = interpolate_dft(xlf, interp1_fs, interp2_fs);
        
        % New sample to be added
        xlf = compact_fourier_coeff(xlf);
        
        % Shift sample
        shift_samp = 2*pi * (pos - sample_pos) ./ (sample_scale * img_support_sz);
        xlf = shift_sample(xlf, shift_samp, kx, ky);
        
        % Init the projection matrix
        projection_matrix = init_projection_matrix(xl, sample_dim, params);
        
        % Project sample
        xlf_proj = project_sample(xlf, projection_matrix);
        
        clear xlw
    elseif params.learning_rate > 0
        if ~params.use_detection_sample
            % Extract image region for training sample
            sample_pos = round(pos);
            sample_scale = currentScaleFactor;
            xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
            
            % Project sample
            xl_proj = project_sample(xl, projection_matrix);
            
            % Do windowing of features
            xl_proj = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_proj, cos_window, 'uniformoutput', false);
            
            % Compute the fourier series
            xlf1_proj = cellfun(@cfft2, xl_proj, 'uniformoutput', false);
            
            % Interpolate features to the continuous domain
            xlf1_proj = interpolate_dft(xlf1_proj, interp1_fs, interp2_fs);
            
            % New sample to be added
            xlf_proj = compact_fourier_coeff(xlf1_proj);
        else
            if params.debug
                % Only for visualization
                xl = cellfun(@(xt) xt(:,:,:,scale_ind), xt, 'uniformoutput', false);
            end
            
            % Use the sample that was used for detection
            sample_scale = sample_scale(scale_ind);
            xlf_proj = cellfun(@(xf) xf(:,1:(size(xf,2)+1)/2,:,scale_ind), xtf_proj, 'uniformoutput', false);
        end
        
        % Shift the sample so that the target is centered
        shift_samp = 2*pi * (pos - sample_pos) ./ (sample_scale * img_support_sz);
        xlf_proj = shift_sample(xlf_proj, shift_samp, kx, ky);
    end
    
    % The permuted sample is only needed for the CPU implementation
    if ~params.use_gpu
        xlf_proj_perm = cellfun(@(xf) permute(xf, [4 3 1 2]), xlf_proj, 'uniformoutput', false);
    end
        
    if params.use_sample_merge
        % Update the samplesf to include the new sample. The distance
        % matrix, kernel matrix and prior weight are also updated
        if params.use_gpu
            [merged_sample, new_sample, merged_sample_id, new_sample_id, distance_matrix, gram_matrix, prior_weights] = ...
                update_sample_space_model_gpu(samplesf, xlf_proj, distance_matrix, gram_matrix, prior_weights,...
                num_training_samples,params);
        else
            [merged_sample, new_sample, merged_sample_id, new_sample_id, distance_matrix, gram_matrix, prior_weights] = ...
                update_sample_space_model(samplesf, xlf_proj_perm, distance_matrix, gram_matrix, prior_weights,...
                num_training_samples,params);
        end
        
        if num_training_samples < params.nSamples
            num_training_samples = num_training_samples + 1;
        end
    else
        % Do the traditional adding of a training sample and weight update
        % of C-COT
        [prior_weights, replace_ind] = update_prior_weights(prior_weights, gather(sample_weights), latest_ind, seq.frame, params);
        latest_ind = replace_ind;
        
        merged_sample_id = 0;
        new_sample_id = replace_ind;
        if params.use_gpu
            new_sample = xlf_proj;
        else
            new_sample = xlf_proj_perm;
        end
    end
    
    if seq.frame > 1 && params.learning_rate > 0 || seq.frame == 1 && ~params.update_projection_matrix
        % Insert the new training sample
        for k = 1:num_feature_blocks
            if params.use_gpu
                if merged_sample_id > 0
                    samplesf{k}(:,:,:,merged_sample_id) = merged_sample{k};
                end
                if new_sample_id > 0
                    samplesf{k}(:,:,:,new_sample_id) = new_sample{k};
                end
            else
                if merged_sample_id > 0
                    samplesf{k}(merged_sample_id,:,:,:) = merged_sample{k};
                end
                if new_sample_id > 0
                    samplesf{k}(new_sample_id,:,:,:) = new_sample{k};
                end
            end
        end
    end

    sample_weights = cast(prior_weights, 'like', params.data_type);
           
    train_tracker = (seq.frame < params.skip_after_frame) || (frames_since_last_train >= params.train_gap);
    
    if train_tracker     
        % Used for preconditioning
        new_sample_energy = cellfun(@(xlf) abs(xlf .* conj(xlf)), xlf_proj, 'uniformoutput', false);
        
        if seq.frame == 1
            % Initialize stuff for the filter learning
            
            % Initialize Conjugate Gradient parameters
            sample_energy = new_sample_energy;
            CG_state = [];
            
            if params.update_projection_matrix
                % Number of CG iterations per GN iteration 
                init_CG_opts.maxit = ceil(params.init_CG_iter / params.init_GN_iter);
            
                hf = cell(2,1,num_feature_blocks);
                proj_energy = cellfun(@(P, yf) 2*sum(abs(yf(:)).^2) / sum(feature_dim) * ones(size(P), 'like', params.data_type), projection_matrix, yf, 'uniformoutput', false);
            else
                CG_opts.maxit = params.init_CG_iter; % Number of initial iterations if projection matrix is not updated
            
                hf = cell(1,1,num_feature_blocks);
            end
            
            % Initialize the filter with zeros
            for k = 1:num_feature_blocks
                hf{1,1,k} = zeros([filter_sz(k,1) (filter_sz(k,2)+1)/2 sample_dim(k)], 'like', params.data_type_complex);
            end
        else
            CG_opts.maxit = params.CG_iter;
            
            % Update the approximate average sample energy using the learning
            % rate. This is only used to construct the preconditioner.
            sample_energy = cellfun(@(se, nse) (1 - params.learning_rate) * se + params.learning_rate * nse, sample_energy, new_sample_energy, 'uniformoutput', false);
        end
        
        % Do training
        if seq.frame == 1 && params.update_projection_matrix
            if params.debug
                projection_matrix_init = projection_matrix;
            end
            
            % Initial Gauss-Newton optimization of the filter and
            % projection matrix.
            if params.use_gpu
                [hf, projection_matrix, res_norms] = train_joint_gpu(hf, projection_matrix, xlf, yf, reg_filter, sample_energy, reg_energy, proj_energy, params, init_CG_opts);
            else
                [hf, projection_matrix, res_norms] = train_joint(hf, projection_matrix, xlf, yf, reg_filter, sample_energy, reg_energy, proj_energy, params, init_CG_opts);
            end
            
            % Re-project and insert training sample
            xlf_proj = project_sample(xlf, projection_matrix);
            for k = 1:num_feature_blocks
                if params.use_gpu
                    samplesf{k}(:,:,:,1) = xlf_proj{k};
                else
                    samplesf{k}(1,:,:,:) = permute(xlf_proj{k}, [4 3 1 2]);
                end
            end
            
            % Update the gram matrix since the sample has changed
            if strcmp(params.distance_matrix_update_type, 'exact')
                % Find the norm of the reprojected sample
                new_train_sample_norm =  0;
                
                for k = 1:num_feature_blocks
                    new_train_sample_norm = new_train_sample_norm + real(gather(2*(xlf_proj{k}(:)' * xlf_proj{k}(:))));% - reshape(xlf_proj{k}(:,end,:,:), [], 1, 1)' * reshape(xlf_proj{k}(:,end,:,:), [], 1, 1));
                end
                
                gram_matrix(1,1) = new_train_sample_norm;
            end
            
            if params.debug
                norm_proj_mat_init = sqrt(sum(cellfun(@(P) gather(norm(P(:))^2), projection_matrix_init)));
                norm_proj_mat = sqrt(sum(cellfun(@(P) gather(norm(P(:))^2), projection_matrix)));
                norm_proj_mat_change = sqrt(sum(cellfun(@(P,P2) gather(norm(P(:) - P2(:))^2), projection_matrix_init, projection_matrix)));
                fprintf('Norm init: %f, Norm final: %f, Matrix change: %f\n', norm_proj_mat_init, norm_proj_mat, norm_proj_mat_change / norm_proj_mat_init);
            end
        else
            % Do Conjugate gradient optimization of the filter
            if params.use_gpu
                [hf, res_norms, CG_state] = train_filter_gpu(hf, samplesf, yf, reg_filter, sample_weights, sample_energy, reg_energy, params, CG_opts, CG_state);
            else
                [hf, res_norms, CG_state] = train_filter(hf, samplesf, yf, reg_filter, sample_weights, sample_energy, reg_energy, params, CG_opts, CG_state);
            end
        end
        
        % Reconstruct the full Fourier series
        hf_full = full_fourier_coeff(hf);
        
        frames_since_last_train = 0;
    else
        frames_since_last_train = frames_since_last_train+1;
    end
    
    % Update the scale filter
    if nScales > 0 && params.use_scale_filter
        scale_filter = scale_filter_update(im, pos, base_target_sz, currentScaleFactor, scale_filter, params);
    end
    
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;
    
    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
    rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
    


    objh{ppp}={thisname,videoPlayer, base_target_sz, block_inds, cam, CG_opts, CG_state, cos_window, currentScaleFactor, debugshow, distance_matrix, feature_dim, feature_extract_info, feature_info, feature_params, feature_sz, feature_sz_cell, features, filter_sz, filter_sz_cell, frames_since_last_train, global_fparams, gram_matrix, hf, hf_full, img_sample_sz, img_support_sz, init_CG_opts, init_target_sz, interp1_fs, interp2_fs, is_color_image, k, k1, kx, ky, latest_ind, max_scale_factor, merged_sample, merged_sample_id, min_scale_factor, new_sample, new_sample_energy, new_sample_id, new_train_sample_norm, nScales, num_feature_blocks, num_training_samples, output_sz, pad_sz, params, pos, ppp, prior_weights, proj_energy, projection_matrix, rect_position_vis, reg_energy, reg_filter, reg_window_edge, res_norms, residuals_pcg, sample_dim, sample_energy, sample_pos, sample_scale, sample_weights, samplesf, scale_filter, scale_step, scaleFactors, scores_fs_feat, search_area, seq, shift_samp, sig_y, target_sz, tracking_result, train_tracker, xl, xlf, xlf_proj, xlf_proj_perm, yf, yf_x, yf_y,to};
end