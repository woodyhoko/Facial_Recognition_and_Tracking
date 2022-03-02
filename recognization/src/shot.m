% make a shot 
function shot(cam)
%Go to target folder
cd('../input');
cam = webcam(1);
cam.Resolution = '1280x720';
data = snapshot(cam);
data = imresize(data, [300 300]);
imwrite(rgb2gray(data), 'target.bmp');
cd('../src');
end
 