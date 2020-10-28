% Do recognition of data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Goal:
%%% Input: 
%%% Output:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function name_str = main (pic)
global face_recog;    % face_recog data
global name_recog;    % Inputed name_recog of face_recog data

isRecog = 0;    % It can be recognized

if isempty(pic)     %To check input
    disp('No input!')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% For Recognition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Normalize selected image
Tr = im2double(pic{5}(:));      % get first picture to do recog.
Ar = Tr-face_recog.mTot;

% Obtain weights of the selected face_recog
Wrec = face_recog.Vf'*face_recog.U'*Ar;

% Obtaining an array of euclidean distances to each face_recog
eDist = [];
for i = 1:face_recog.M
    eDist = [eDist sqrt(( norm( Wrec - face_recog.Wf(:,i)) )^2)]; 
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
    if mod(i, 5) == 0
        caseDis(caseIndex) = sum/5;
        if minDis > caseDis(caseIndex)
            minDis = caseDis(caseIndex);
            minIndex = caseIndex;
        end
        sum = 0;
        caseIndex = caseIndex + 1; 
    end
end    

caseDis
name_recog


if size(name_recog,2)==1
    t =1;
elseif size(name_recog,2)==2
    t =20;
elseif size(name_recog,2)==3
    t=50;
else
    t=500;
end

if minDis < t      % Threshold 
    isRecog = 1;
else
    isRecog = 0;
end

if isRecog == 1
    Matching_case = minIndex;
    
    % Deal with string 
    name_str = name_recog{Matching_case};  

    display(name_str);
    return;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% For training data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else 
    % Go to train data, because no match
    % Input the name_recog of picture
    face_recog.k = face_recog.k + 1;
    k = face_recog.k;
    
    prompt={'Please enter your name'};
            nn = '';
            defaultans = {'Unknow'};
            options.Interpreter = 'tex';
            answer = inputdlg(prompt,nn,[1 40],defaultans,options);
            answer = char(answer);
            
            name_recog{size(name_recog,2)+1} = answer;
    
    % Initialize the face_recog.data
    if face_recog.data == 0
        face_recog.data = im2double(pic{1}(:));
        for i=2:5
            face_recog.data = cat(4, face_recog.data, im2double(pic{i}(:)));
        end
    else
        for i=1:5
            face_recog.data = cat(4, face_recog.data, im2double(pic{i}(:)));
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Principal Component Analysis (PCA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    display('Loading please wait...')
    % Read images in T matrix
    [nRow nCol M] = size(face_recog.data);
    face_recog.M = M;
    
    % T is a matrix containing the reshaped vectors for each image
    T = reshape(face_recog.data,[nRow*nCol M]);
    
    % mTot is the mean of the entire set of training images
    mTot = mean(T,2);
    face_recog.mTot = mTot;
    
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
    face_recog.U = U;

    % Obtaining PCA weights
    Wpca = U'*A;    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Fisher's Linear Discriminant Analysis
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    cMean = zeros(M-k,M-k);
    Sb = zeros(M-k,M-k);
    Sw = zeros(M-k,M-k);
    
    pcaMean = mean(Wpca,2);
    
    for i = 1:k
        cMean = mean(Wpca(:,5*i-(5-1):5*i),2);
        Sb = Sb + (cMean-pcaMean)*(cMean-pcaMean)';
    end    
    
   Sb = 5*Sb;

    for i = 1:k
        cMean = mean(Wpca(:,5*i-(5-1):5*i),2);
        for j = 5*i-(5-1):5*i
             Sw = Sw + (Wpca(:,j)-cMean)*(Wpca(:,j)-cMean)';
        end
    end

    % Obtaining Fisher eigenvectors and eigenvalues
    [Vf, Df] = eig(Sb,Sw);

    % Calculating weights
    Df = fliplr(diag(Df));
    Vf = fliplr(Vf);
    face_recog.Vf = Vf;

    % Calculating fisher weights
    Wf = Vf'*Wpca;
    face_recog.Wf = Wf;
    
    name_str = name_recog{end};
end
end
