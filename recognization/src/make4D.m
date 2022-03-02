% To load the data into 4D.mat
function face =  make4D(case_num, pic_num)
face = zeros();
display('Loading the input images...')
for pid = 1:case_num
    path_prot = ['../input/sample/' int2str(pid) '/'];
    if pid == 1
        path = [path_prot int2str(1) '.bmp'];
        input = im2double(imread(path)); 
        face = input;
        for i = 2:pic_num
            path = [path_prot int2str(i) '.bmp'];
            input = im2double(imread(path));
            face = cat(4, face, input);
        end
    else
        for i = 1:pic_num
            path = [path_prot int2str(i) '.bmp'];
            input = im2double(imread(path));
            face = cat(4, face, input);
        end
    end
end
end