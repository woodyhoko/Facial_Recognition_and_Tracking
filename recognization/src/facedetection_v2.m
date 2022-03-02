clear;
clc;
close all

faceDetector = vision.CascadeObjectDetector();

%Get the input device using image acquisition toolbox,resolution = 640x480 to improve performance
obj =imaq.VideoDevice('winvideo', 1, 'MJPG_320x240','ROI', [1 1 320 240]);
set(obj,'ReturnedColorSpace', 'rgb');
figure('menubar','none','tag','webcam');

bbox = [];
% the currCase of the sample
currCase = 1;

% the src location
path_src = 'C:\Users\user\Desktop\ECO_project\ECOcode\ECO-master\recognization\src';

while (true)
    frame=step(obj);
    
    bbox_h = size(bbox, 1);
    
    bbox=step(faceDetector,frame);
    
    % makeVideo with bbox 
%     if size(bbox, 1) > bbox_h
%         release(obj);
%         currCase = makeVideo(5, bbox, path_src);
%         % recall the obj
%         obj =imaq.VideoDevice('winvideo', 1, 'MJPG_320x240','ROI', [1 1 320 240]);
%         set(obj,'ReturnedColorSpace', 'rgb');
%         figure('menubar','none','tag','webcam');
%     end
    
    boxInserter  = vision.ShapeInserter('BorderColor','Custom',...
    'CustomBorderColor',[255 255 0]);
    videoOut = step(boxInserter, frame,bbox);
   
    imshow(videoOut,'border','tight');
    
    f = findobj('tag','webcam');
    
    if (isempty(f));
        [hueChannel,~,~] = rgb2hsv(frame);

% Display the Hue Channel data and draw the bounding box around the face.
figure, imshow(hueChannel), title('Hue channel data');


rectangle('Position',bbox,'EdgeColor','r','LineWidth',1)
hold off
noseDetector = vision.CascadeObjectDetector('Nose');
faceImage    = imcrop(frame,bbox);
imshow(faceImage)  
noseBBox     = step(noseDetector,faceImage);

noseBBox(1:1) = noseBBox(1:1) + bbox(1:1);
videoInfo    = info(obj);
ROI=get(obj,'ROI');
VideoSize = [ROI(3) ROI(4)];

videoPlayer  = vision.VideoPlayer('Position',[300 300 VideoSize+30]);
tracker = vision.HistogramBasedTracker;
initializeObject(tracker, hueChannel, bbox);

while (1)
    
% Extract the next video frame
    frame = step(obj);
% RGB -> HSV
    [hueChannel,~,~] = rgb2hsv(frame);

    % Track using the Hue channel data
    bbox = step(tracker, hueChannel);

    % Insert a bounding box around the object being tracked
    videoOut = step(boxInserter, frame, bbox);

    %Insert text coordinates

    % Display the annotated video frame using the video player object
    step(videoPlayer, videoOut);
    pause (.2)
end

% Release resources
release(obj);
release(videoPlayer);

        close(gcf)
        break
    end
    pause(0.05)
end
release(obj)
