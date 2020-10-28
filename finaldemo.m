function finaldemo()
clear all;
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

global objh;
global img;

csc=1;

faceDetector=vision.CascadeObjectDetector('FrontalFaceCART'); %Create a detector object
setup_paths();
totaldebug=csvread("321654\\tt.txt");
debugshow=totaldebug(1);
debuginfo=totaldebug(2);

video_path = '321654\\321654';%simular color
seq= load_video_info(video_path);
videoPlayer  = vision.VideoPlayer('Position',[100 100 [1280, 720]+30]);
a=1;
% Run ECO
cam = webcam(1);
cam.Resolution = '1280x720';
objh={};
n=0;
tempcc=[0 0 5];
tempccc=cell(5);
%preview(cam)
while(true)
    tic;
    img = snapshot(cam);
    img=flip(img,2);
    if mod(n,60)==0 || tempcc(3)~=5
        BB=step(faceDetector,img);
        cccc = 0;
        for nn=1:size(BB,1)
            checkb=0;
            for nkn=1:size(objh,2)
                ttpp=cell2mat(objh{nkn}(57));
                if abs(ttpp(1)-BB(nn,1))<80 && abs(ttpp(2)-BB(nn,2))<80
                    checkb=1;
                    objh{nkn}{end}=0;
                    objh{nkn}{57}=BB(nn,1:end);
                    break;
                end
            end
            
            if checkb==0 && (BB(nn,3)>100 && BB(nn,4)>100 && BB(nn,3)<350 && BB(nn,4)<350 && tempcc(3)~=0 && abs(tempcc(1)-BB(nn,1))<80 && abs(tempcc(2)-BB(nn,2))<80)
                tempcc(1)=BB(nn,1);
                tempcc(2)=BB(nn,2);
                tempcc(3)=tempcc(3)-1;
                x1 = BB(nn, 1);
                y1 = BB(nn, 2);
                x2 = BB(nn, 1)+BB(nn, 3);
                y2 = BB(nn, 2)+BB(nn, 4);
                img2 = img(y1:y2,x1:x2);
                %imgtt = rgb2gray(img2);
                img3 = imresize(img2, [150 150]);
                tempccc{tempcc(3)+1}=img3;
                img=insertObjectAnnotation(img, 'rectangle', BB(nn,1:end), tempcc(3));
                cccc = 1;
            elseif checkb==0 && (BB(nn,3)>150 && BB(nn,4)>150 && BB(nn,3)<350 && BB(nn,4)<350) && abs(tempcc(1)-BB(nn,1))<80 && abs(tempcc(2)-BB(nn,2))<80
                objh{size(objh,2)+1}=testing_ECO_HC(seq,debugshow,debuginfo,BB(nn,1:end),cam,videoPlayer);
                track_r(size(objh,2),1);
                %main(tempccc{5});
                objh{end}{1}=main(tempccc);
                for ii=1:size(objh,2)-1
                    if strcmp(objh{ii}{1},objh{end}{1})
                        objh(ii)=[];
                        break;
                    end
                end
                cccc = 1;
            elseif checkb==0 && (BB(nn,3)>150 && BB(nn,4)>150 && BB(nn,3)<350 && BB(nn,4)<350)
                tempcc(1)=BB(nn,1);
                tempcc(2)=BB(nn,2);
                tempcc(3)=4;
                x1 = BB(nn, 1);
                y1 = BB(nn, 2);
                x2 = BB(nn, 1)+BB(nn, 3);
                y2 = BB(nn, 2)+BB(nn, 4);
                img2 = img(y1:y2,x1:x2);
                %imgtt = rgb2gray(img2);
                img3 = imresize(img2, [150 150]);
                tempccc{tempcc(3)+1}=img3;
                img=insertObjectAnnotation(img, 'rectangle', BB(nn,1:end), tempcc(3));
                cccc = 1;
            end
        end
        if cccc == 0
            tempcc(3)=5;
            tempccc=cell(5);
        end
    end
    
    n=n+1;
    tempp={};
    
    if size(objh,2)~=0          %udate only one objh
        track_r(mod(n,size(objh,2))+1,0);
    end
    
%     for sis=1:size(objh,2)    %update every objh
%         track_r(sis,0);
%     end
    
    for gn=size(objh,2):-1:1
        objh{gn}{end}=objh{gn}{end}+1;
        if objh{gn}{end}==250 || objh{gn}{57}(1)<20 || objh{gn}{57}(2)<20 || objh{gn}{57}(1)+objh{gn}{57}(3)>1280 || objh{gn}{57}(2)+objh{gn}{57}(4)>720
            objh(gn)=[];
        else
            img=insertObjectAnnotation(img, 'rectangle', objh{gn}{57}, objh{gn}{1},'Color',{'green'});
        end
    end
    
    %track_v(tempp,videoPlayer,img);
    %im_to_show = img;
    img=insertText(img,[10 10], num2str(1/to));
    %rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
    step(videoPlayer, img);
    
end