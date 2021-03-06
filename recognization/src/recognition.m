% Do the Face Recognization
% Clear memory and console
% close all
% clear
% clc
function recognition(ki, ni)
% Define the  groundtruth path 
%path_gt = ['C:\\Users\\user\\Desktop\\ECO_project\\ECOcode\\testing\\testData\\groundtruth_rect.txt'];

%Define the paths
path_src = ['C:\Users\user\Desktop\ECO_project\ECOcode\ECO-master\recognization\src'];

%Define variables
 k = 2;        % Number of classes
 n = 5;        % Number of images per class

% Start to input sample
% display('First, input sample database: ')
%isInput = input('Do you wanna to input sample? \n(0)No \n(1)Yes \n');
%isInput = 1;

% Capture the sample file
% if isInput == 1
%     k = makeVideo(n);
% end

% cd (path_src);
% Input the target
%display('Second, input the target: ')
%isInput = input('Do you wanna to input target? \n(0)No \n(1)Yes \n');

% Capture the target file
% if isInput == 1
%     shot;
% end

cd('../src');
% To load the data into face
face = make4D(k, n);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Principal Component Analysis (PCA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Loading please wait...')

% Read images in T matrix
[nRow nCol M] = size(face);

% T is a matrix containing the reshaped vectors for each image
T = reshape(face,[nRow*nCol M]);

% mTot is the mean of the entire set of training images
mTot = mean(T,2);

% substract mean
A = T-repmat(mTot,1,M);

% Obtaining eigenvalues and eigenvectors of A'A
[V,D] = eig(A'*A);

% Obtaning more relevant eigenvalues and eigenvectors
eval = diag(D);

peval = [];
pevec = [];

for i = M:-1:k+1
    peval = [peval eval(i)];
    pevec = [pevec V(:,i)];
end

% Obtaining the eigenvectors
U = A * pevec; 

% Obtaining PCA weights
 Wpca = U'*A;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fisher's Linear Discriminant Analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
% Obtaining Sb and Sw
cMean = zeros(M-k,M-k);
Sb = zeros(M-k,M-k);
Sw = zeros(M-k,M-k);

pcaMean = mean(Wpca,2);

for i = 1:k
    cMean = mean(Wpca(:,n*i-(n-1):n*i),2);
    Sb = Sb + (cMean-pcaMean)*(cMean-pcaMean)';
end

Sb = n*Sb;

for i = 1:k
    cMean = mean(Wpca(:,n*i-(n-1):n*i),2);
    for j = n*i-(n-1):n*i
         Sw = Sw + (Wpca(:,j)-cMean)*(Wpca(:,j)-cMean)';
    end
end

% Obtaining Fisher eigenvectors and eigenvalues
[Vf, Df] = eig(Sb,Sw);

% Calculating weights
 Df = fliplr(diag(Df));
 Vf = fliplr(Vf);

% Calculating fisher weights
Wf = Vf'*Wpca;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SELECT FACE BASED ON THE SHORTEST EUCLIDEAN DISTANCE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% Normalize selected image
input = imread('../input/target.bmp');
%input = imresize(input, [300 300]);
Tr = im2double(input(:));
Ar = Tr-mTot;

% Obtain weights of the selected face
Wrec = Vf'*U'*Ar;

temp = 0;

% Obtaining an array of euclidean distances to each face
eDist = [];
for i = 1:M
    eDist = [eDist sqrt(( norm( Wrec - Wf(:,i)) )^2)]; 
end

% Find minimum distance and the corresponding index
minDis = 999999;
minIndex = 0;
% The distance per case
caseDis = [];
sum = 0;
caseIndex = 1;

% Compute the distance of every cases, summing up the distances per
% classes, and compare each cases. Find the min distance of the cases.
for i = 1:length(eDist)
    sum = eDist(i) + sum;
    if mod(i, n) == 0
        caseDis(caseIndex) = sum/n;
        if minDis > caseDis(caseIndex)
            minDis = caseDis(caseIndex);
            minIndex = caseIndex;
        end
        sum = 0;
        caseIndex = caseIndex + 1; 
    end
end    

Matching_case = minIndex;

% Matching index
display(Matching_case);
end

    





