img = imread('Lenna.png');
img = rgb2gray(img);
img = double(img) ./ 255.0;
img(img < 0.5) = 0;
img(img > 0.5) = 1;

t = tensor(img);
m = cp_nmu(t, 30);

m = double(m);
tmp = max(m(:));
org_m = m ./ tmp;

m(m < 0.5) = 0;
m(m > 0.5) = 1;

imshow(org_m);