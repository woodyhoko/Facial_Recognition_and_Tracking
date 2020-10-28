cam = webcam();
cam.Resolution = '640x360';
videoPlayer  = vision.VideoPlayer('Position',[100 100 [1280 720]+30]);
while 0==0
    tic;
    I = flip(snapshot(cam),2);
    fps = 1/ toc;
    I = insertText(I,[10,20],['fps = ', num2str(fps)]);
    step(videoPlayer, I);

end