function pic2avi()
%fileNames = dir(fullfile('D:\ECO-master\ECO-master\sequences\Deer\Deer\img','*.jpg'));

%Create Video with Image Sequence
clear all
clc

%Make the Below path as the Current Folder
cd('D:\ECO-master\ECO-master\sequences\Deer\Deer\img');

%Obtain all the JPEG format files in the current folder
Files = dir('*.jpg');

%Number of JPEG Files in the current folder
NumFiles= size(Files,1);


%To write Video File
VideoObj = VideoWriter('deer.avi');
%Number of Frames per Second
VideoObj.FrameRate = 50;
%Define the Video Quality [ 0 to 100 ]
VideoObj.Quality = 80;

%Open the File 'Create_video01.avi'
open(VideoObj);

for i = 1 : NumFiles 
%Read the Image from the current Folder
I = imread(Files(i).name);

ResizeImg = imresize(I,[432 576]);

%Convert Image to movie Frame
frame = im2frame(ResizeImg);

%Each Frame is written five times.
for j = 1 : 5
%Write a frame 
writeVideo(VideoObj, frame);
end

end


%Close the File 'Create_Video01.avi
close(VideoObj);