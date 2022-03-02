function startdemo(seq,x,y,w,h,de)
    a=fopen(seq+'\test.txt','w');
    fprintf(a,'%d,%d,%d,%d ',x,y,w,h);
    
end