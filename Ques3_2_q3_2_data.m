
data = load('/Users/mahimaparashar/Desktop/GradSchool/Machine Learning/Homework2/hw2data/q3_1_data.mat');
[d, n] = size(data.trD);
trD = data.trD';
trLb = data.trLb;

X_val = data.valD;
Y_val = data.valLb;
[d_val, n_val] = size(X_val);
%for ques 3.6
% trLb(trLb == -1) = 2;

data = [trD trLb];
classes = unique(trLb);
classes = sort(classes);
k = size(classes, 1);
C = 10;

max_epoc = 200;
eta_0 = 1;
eta_1 = 100; 
L = zeros(d,n);

Y_hat = zeros(n,1);
W = zeros(d,k);
loss = zeros(max_epoc,1);


for epoc = 1: max_epoc
 
    eta = eta_0 /(eta_1 + epoc);
    shuffled_data = data(randperm(size(data,1)), :);
    [n, d] = size(shuffled_data);
    X = shuffled_data(1:end, 1:d-1)';
    Y = shuffled_data(1:end, d);
    losstemp = zeros(n,1);
    temp = zeros(k,1);
    
    [M, I] = maxk(W'*X,2,1);
    
    for i = 1:n
        %calculating Y_hat
        if(I(1, i) == Y(i))
           Y_hat(i) = classes(I(2,i));
        else
           Y_hat(i) = classes(I(1,i));
        end

        Lvalue = max(W(:,Y_hat(i))'*X(:,i) - W(:, Y(i))'*X(:,i) + 1, 0);
        
        for j = 1:k
            if(Lvalue > 0)
                if(j == Y(i))
                    L(:,i) = W(:,j)/n + C*-1*X(:,i); 
                elseif(j == Y_hat(i))
                    L(:,i) = W(:,j)/n + C*X(:,i);
                else
                   L(:,i) = W(:,j)/n;
                end
            else
                L(:,i) =  W(:,j)/n;
            end
            W_curr = W(:,j);
            W(:,j)  = W_curr - eta*L(:,i);
            temp(j,1) = W(:,j)'*W(:,j);
        end
        
        losstemp(i,1) = sum(temp)/(2*n) + C*max(W(:,Y_hat(i))'*X(:,i) - W(:, Y(i))'*X(:,i) + 1, 0);
    end
    loss(epoc, 1) = sum(losstemp);
    epoc
end

R = W'*X;
Y_pred = zeros(n, 1);

for i = 1:n
    [m, I] = max(R(:,i));
    Y_pred(i) = classes(I);
end

temp = Y_pred == Y;
acc = sum(temp)/n * 100;

T = W'*X_val;
Yval_pred = zeros(size(Y_val,1), 1);

for i = 1:n_val
    [m, I] = max(T(:,i));
    Yval_pred(i) = classes(I);
end

tempval = Yval_pred == Y_val;
acc1 = sum(tempval)/n_val * 100;

figure
epoc = (1:max_epoc)';
plot(epoc, loss);
title('Loss/epoc Plot');
xlabel('epoc');
ylabel('Loss');

