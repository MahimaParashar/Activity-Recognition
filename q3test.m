data = load('/Users/mahimaparashar/Desktop/GradSchool/Machine Learning/Homework2/hw2data/q3_2_data.mat');
X_test = data.tstD;
[d_test , n_test] = size(X_test);
T_test = W'*X_test;
Y_test = zeros(n_test, 1);

for i = 1:n_test
    [m_test, I_test] = max(T_test(:,i));
    Y_test(i) = classes(I_test);
end

csvwrite('q3_kaggle.csv', Y_test);
k