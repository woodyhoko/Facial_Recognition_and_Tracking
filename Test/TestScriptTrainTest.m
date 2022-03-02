aTrain = {};
bTrain = {};
global face_recog;    % face_recog data
global name_recog;    % Inputed name_recog of face_recog data

face_recog.data = zeros();
face_recog.M = 1;
face_recog.k = 0;     % # of classes
face_recog.mTot = 0;  % mean value
face_recog.Vf = 0;    % Fisher eigenvectors
face_recog.U = 0;     % PCA's eigenvectors
face_recog.Wf = 0;    % fisher weight
% 
name_recog = [];
for i = 1:5
    img = imread(['D:\ECO-master\ECO-master\Test\a\' num2str(i+3) '.bmp']);
    aTrain{i} = cat(3,img,img,img);
end

for i = 1:5
    img = imread(['D:\ECO-master\ECO-master\Test\b\' num2str(i+3) '.bmp']);
    bTrain{i} = cat(3,img,img,img); 
end

aTest1 = imread(['D:\ECO-master\ECO-master\Test\a\18.bmp']);
aTest2 = imread(['D:\ECO-master\ECO-master\Test\a\19.bmp']);

bTest1 = imread(['D:\ECO-master\ECO-master\Test\b\18.bmp']);
bTest2 = imread(['D:\ECO-master\ECO-master\Test\b\19.bmp']);

main(aTrain);
main(bTrain);

main({[],[],[],[],cat(3,aTest1,aTest1,aTest1)})%aaa
main({[],[],[],[],cat(3,aTest2,aTest2,aTest2)})%aaa
main({[],[],[],[],cat(3,bTest1,bTest1,bTest1)})%bbb
main({[],[],[],[],cat(3,bTest2,bTest2,bTest2)})%bbb