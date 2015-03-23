load mimic2-tensor-data;

[t, u] = cp_nmu(phe_tensor, 30);

R = 30;
mat_A = u{1};
mat_B = u{2};
mat_C = u{3};

save('mimic2-decomp-cp-30.mat', 'mat_A', 'mat_B', 'mat_C', 'R');