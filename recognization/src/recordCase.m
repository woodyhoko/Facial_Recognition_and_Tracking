% Write Data
function recordCase(num, name_str)
%cd ..
%cd input
%if exist('name.txt', file) ~= 2
    
fileID = fopen('name.txt', 'w');
for i=1:num
   fprintf(fileID, '%s\n', name_str); 
end
fclose(fileID);
%cd ../src
end