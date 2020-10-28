function demo_ECO_HC()
% This demo script runs the ECO tracker with hand-crafted features on the
% included "Crossing" video.

% Add paths
setup_paths();
totaldebug=csvread("C:\\321654\\tt.txt");
debugshow=totaldebug(1);
debuginfo=totaldebug(2);
% Load video information
%video_path = 'sequences/crossing';
%video_path = 'sequences/singer2/singer2';%failed
%video_path = 'sequences/Basketball/Basketball';%default
%video_path = 'sequences/BlurBody/BlurBody';%shake
%video_path = 'sequences/Freeman4/Freeman4';%noise
%video_path = 'sequences/Couple/Couple';%simular
%video_path = 'sequences/Dudek/Dudek';%glass
video_path = 'C:\\321654\\321654';%simular color
seq= load_video_info(video_path);

% Run ECO
results = testing_ECO_HC(seq,debugshow,debuginfo);