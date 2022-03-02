clear;
faceDetector=vision.CascadeObjectDetector('FrontalFaceCART'); %Create a detector object
cam = webcam(1);
cam.Resolution = '1280x720';
while(true)
    img = snapshot(cam); %Read input image

    img=rgb2gray(img); % convert to gray

    BB=step(faceDetector,img); % Detect faces

    iimg = insertObjectAnnotation(img, 'rectangle', BB, 'Face'); %Annotate detected faces.

    figure(1);
    imshow(iimg); 
    title('Detected face');

end

%                   data = snapshot(cam);
%                 img = rgb2gray(data);
%                 % The image revolution
%                 img = imresize(img, [720 1280]);
%                 %img = imcrop(img, [bb(i, 1) bb(i, 2) bb(i, 3) bb(i, 4)]);
%                 x1 = bb(i, 1);
%                 y1 = bb(i, 2);
%                 x2 = bb(i, 1)+bb(i, 3);
%                 y2 = bb(i, 2)+bb(i, 4);
%                 img2 = img(y1:y2,x1:x2);
%                 figure(2)
%                 imshow(img2)
%                 img = imresize(img2, [300 300]);