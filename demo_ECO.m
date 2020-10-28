
% This demo script runs the ECO tracker with deep features on the
% included "Crossing" video.

% Add paths
setup_paths();

% Load video information
%video_path = 'sequences/Crossing';
%video_path = 'sequences/singer2/singer2';%failed
video_path = 'sequences/Basketball/Basketball';%default
[seq, ground_truth] = load_video_info(video_path);

% Run ECO
results = testing_ECO(seq);