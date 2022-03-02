clear;
clc;
close all

faceDetector=vision.CascadeObjectDetector('FrontalFaceCART'); %Create a detector object
cam = webcam(1);
cam.Resolution = '1280x720';
BB = [];
run = 1;

while(true)
    img = snapshot(cam); %Read input image

    img = rgb2gray(img); % convert to gray

    BB = step(faceDetector,img); % Detect faces
    
    bb_h = size(BB, 1);
    
    CC_ind = 1;
    CC = [];
    %%% Need fix
    for i=1:bb_h
        if BB(i, 4) > 300
            %CC(CC_ind, :) = BB(i, 1:end);
            %CC_ind = CC_ind+1;
            BB(i, 1:end);
        end            
    end
   
    %clear(BB);
    %BB =CC;
    % the src location
    path_src = 'C:\Users\user\Desktop\ECO_project\ECOcode\ECO-master\recognization\src';
    currCase = makeVideo(5, BB, path_src, cam, 'AA');
    iimg = insertObjectAnnotation(img, 'rectangle', BB, 'Face'); %Annotate detected faces.

    figure(1);
    imshow(iimg); 
    %preview(cam)
    title('Detected face');
    close(gcf)
    break
end