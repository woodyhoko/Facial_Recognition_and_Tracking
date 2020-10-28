% Record the face && build the test data
% Output: the number of cases
function currCase = makeVideo(n, bb, path_src, cam, name_str)
% Compute the currCase
currCase = size(bb, 1);

cd(path_src);
cd( '../input');
% Reset the folder
% if exist('sample', 'dir')
%     rmdir sample s;
% end
mkdir('sample');
cd('sample');

        % Create a folder to save pictures
        for i=1:size(bb, 1)
            mkdir(int2str(i));
            cd(int2str(i));
            % Until intercepting 100 frames, then stopping
            picNum = 1;
            %while (vid.FramesAcquired<=(n-1))
            for j=1:n
                %display(vid.FramesAcquired)
                data = snapshot(cam);
                img = rgb2gray(data);
                % The image revolution
                img = imresize(img, [720 1280]);
                %img = imcrop(img, [bb(i, 1) bb(i, 2) bb(i, 3) bb(i, 4)]);
                x1 = bb(i, 1);
                y1 = bb(i, 2);
                x2 = bb(i, 1)+bb(i, 3);
                y2 = bb(i, 2)+bb(i, 4);
                img2 = img(y1:y2,x1:x2);
                figure(2)
                imshow(img2)
                img = imresize(img2, [300 300]);
                imwrite(img, [int2str(picNum), '.bmp']);
                picNum = picNum + 1;
            end
            cd(path_src);
            recordCase(i, name_str);
            cd ../input/sample
        end
        
        %stop(vid)
        cd(path_src);
end