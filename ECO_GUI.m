function varargout = ECO_GUI(varargin)
%ECO_GUI MATLAB code file for ECO_GUI.fig
%      ECO_GUI, by itself, creates a new ECO_GUI or raises the existing
%      singleton*.
%
%      H = ECO_GUI returns the handle to a new ECO_GUI or the handle to
%      the existing singleton*.
%
%      ECO_GUI('Property','Value',...) creates a new ECO_GUI using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to ECO_GUI_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      ECO_GUI('CALLBACK') and ECO_GUI('CALLBACK',hObject,...) call the
%      local function named CALLBACK in ECO_GUI.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ECO_GUI

% Last Modified by GUIDE v2.5 28-Dec-2017 19:20:56

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ECO_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @ECO_GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
   gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before ECO_GUI is made visible.
function ECO_GUI_OpeningFcn(hObject, ~, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDviATA)
% varargin   unrecognized PropertyName/PropertyValue pairs from the
%            command line (see VARARGIN)

% Choose default command line output for ECO_GUI
handles.output = hObject;

%%% mac OSX
% axes(handles.screen);
% vid =webcam('gent1',1);
% hImage=image(zeros(720,1280,3),'Parent',handles.screen);
% preview(vid,hImage);

%%% windows
vid=videoinput('winvideo'); 
preview(vid);
%data = getsnapshot(vid);
set(vid,'LoggingMode','memory') 
set(vid,'FramesPerTrigger',1)%設定開始擷取影像時擷取1張 
triggerconfig(vid, 'Manual')%設定手動開始 



% Update handles structure
guidata(hObject, handles);

% UIWAIT makes ECO_GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = ECO_GUI_OutputFcn(~, ~, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in capture.
function capture_Callback(~, ~, ~)
% hObject    handle to capture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%------------- 
vid = videoinput('winvideo', 1);
start(vid)
%trigger(vid)
[data , ~]=getdata(vid,10);%擷取到的影像放入data陣列中 
for i=1:10 
    k=int2str(i); 
    imname=['webpic',k,'.bmp']; 
    imwrite(data(:,:,:,i),imname); 
end 
stop(vid);%停止物件動作 
%-------- 
delete(vid); 
clear vid; 
%--------- 
