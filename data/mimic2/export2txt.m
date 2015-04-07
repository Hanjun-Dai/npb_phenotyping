load mimic2-tensor-data;

fid = fopen('mimic2-triplets.txt', 'w');

fprintf(fid, '%d %d %d %d\n', length(A), max(A), max(C), max(B));
p = randperm(length(A));

for i = 1 : length(A)
    fprintf(fid, '%d %d %d\n', A(p(i)) - 1, C(p(i)) - 1, B(p(i)) - 1);
end

fclose(fid);