data = load('/Users/mahimaparashar/Desktop/GradSchool/Machine Learning/Homework2/hw2data/q3_1_data.mat');
[d, n] = size(data.trD);
X = data.trD;
Y = data.trLb;
X_val = data.valD;
Y_val = data.valLb;

C = 10;
H = (Y*Y').*(X'*X);
f = -1*ones(1,n);
Aeq = Y';
Beq = 0;
A = zeros(1,n);
b = 0;

lb = zeros(n, 1);
ub = C * ones(n,1);

alpha = quadprog(H, f, A, b, Aeq,Beq,lb, ub);
w = X * (alpha .* Y);
i = min(find((alpha >= 0) & (Y == 1)));
T = X'*X;
b = 1 - T(i,:)*(alpha.*Y);

Y_pred_train = X'*w + b;
Y_pred_train(Y_pred_train >= 1) = 1;
Y_pred_train(Y_pred_train < 1) = -1;
train_class_diff = Y_pred_train - Y;

Y_pred = X_val'*w + b;
Y_pred(Y_pred >= 1) = 1;
Y_pred(Y_pred < 1) = -1;
accuracy = sum(Y_pred == Y_val)/size(Y_val,1);

HW2_Utils.genRsltFile(w, b, "val", "outputFile");

hinge_loss = 1 - Y_val' * (X_val'*w + b);
hinge_loss(hinge_loss < 0) = 0;
Objective = w'*w/2 + C* sum(hinge_loss);

ConfusionMatrix = confusionmat(Y_val, Y_pred);
ConfusionMatrix_train = confusionmat(Y, Y_pred_train);
numberOfSV = ConfusionMatrix_train(1,2) + ConfusionMatrix_train(2,1) + sum((alpha < 0.01) & train_class_diff == 0);







