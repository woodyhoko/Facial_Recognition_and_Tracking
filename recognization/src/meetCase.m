% Meet the case
function name = meetCase(num)
cd ..
cd input
fileID = fopen('name.txt', 'r');
readFile = fscanf(fileID, '%s');
name = readFile(num);
fclose(fileID);
cd ../src
end