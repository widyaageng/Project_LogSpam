clear;clc;
spamData = load('spamData.mat');

%traindata
r_data_feat = spamData.Xtrain;
r_data_label = spamData.ytrain;
r_bin_dat = zeros(size(r_data_feat,1),size(r_data_feat,2));
N = size(r_data_feat,1);


%testdata
t_data_feat = spamData.Xtest;
t_data_label =spamData.ytest;
t_bin_dat = zeros(size(t_data_feat,1),size(t_data_feat,2));


%-------------------------------------------------
% preprocessing
%-------------------------------------------------

%--Feature Binarization--
bin_threshold = 0;
%traindata binarization with treshold
for i=1:size(r_data_feat,2)
    r_bin_dat(:,i) = double(imbinarize(r_data_feat(:,i),bin_threshold));
end

%testdata binarization with treshold
for i=1:size(t_data_feat,2)
    t_bin_dat(:,i) = double(imbinarize(t_data_feat(:,i),bin_threshold));
end

%-------------------------------------------------
% estimating prior probability of class/label using MLE
%-------------------------------------------------

% # of data from r_data_feat that belongs to class 1 (spam)
Nclass_1 = 0;

%2by57 matrix containing # of appearance of feature '1' in each class
%class c is on (c+1)th row, jth feature is on jth column
Nfeat1_mat = zeros(2,size(r_data_feat,2));

%2by57 matrix containing # of appearance of feature '0' in each class
%class c is on (c+1)th row, jth feature is on jth column
Nfeat0_mat = zeros(2,size(r_data_feat,2));

for i=1:size(r_bin_dat,1)
    if r_data_label(i)>0.5 %to prevent floating issue at 'near' 0 or 1
        Nclass_1 = Nclass_1 + 1;
    end
    
    for j=1:length(r_bin_dat(i,:))
        if r_bin_dat(i,j)>0.5 %to prevent floating issue at 'near' 0 or 1
            Nfeat1_mat(r_data_label(i)+1,j)= Nfeat1_mat(r_data_label(i)+1,j) + 1;
        else
            Nfeat0_mat(r_data_label(i)+1,j)= Nfeat0_mat(r_data_label(i)+1,j) + 1;
        end
    end
end

% MLE prior probability of class/label
prior_pi_one = Nclass_1/size(r_bin_dat,1);
prior_pi_zero = 1 - prior_pi_one;

% set of probability of feature jth appearing (getting 1) in spam data (class 1)
% first element correspond to probability of 1st feature having '1' in spam data (class 1)
% and so on for 2nd, 3rd, .... jth which is 57th feature

%-------------------------------------------------
% alpha iteration to obtain feature probability theta;
% each row of theta matrix correspond to alpha value (identified by a)
% and b-th column (identified with variable b) corresponds to feature b-th;
% so, theta_class1_(a,b) contains probability of feature b-th equals to 1
% with alpha value a in beta distribution posterior mean;
% and theta_class0_(a,b) containes probability of feature b-th equals to 0
% with alpha value a in beta distribution posterior mean;
%-------------------------------------------------

alpha = 0:0.5:100;
theta_class1_beta = zeros(size(alpha,1),size(Nfeat1_mat,2));
theta_class0_beta = zeros(size(alpha,1),size(Nfeat0_mat,2));
for a=1:length(alpha)
    
    % iteration over feature number, size(r_data_feat,2) = 57
    for b=1:size(r_data_feat,2)
        theta_class1_beta(a,b) = (Nfeat1_mat(2,b)+alpha(a))/(Nclass_1+(2*alpha(a)));
        theta_class0_beta(a,b) = (Nfeat0_mat(1,b)+alpha(a))/(N-Nclass_1+(2*alpha(a)));
    end
end

%------------------------------------------------------------------------
% testing on training data/validation and calculating training error rate
%------------------------------------------------------------------------

r_test_label = 0*r_data_label;
pone = zeros(1,length(r_test_label));
pzero = zeros(1,length(r_test_label));
train_error = zeros(1,length(alpha));

for a=1:length(alpha)
    for i=1:length(r_test_label)
        pone(i) = log(prior_pi_one) + (r_bin_dat(i,:)*(log(theta_class1_beta(a,:)))') + ((1-r_bin_dat(i,:))*(log(1-theta_class1_beta(a,:)))');
        %pone(i) = exp(pone(i));
        pzero(i) = log(prior_pi_zero) + ((1-r_bin_dat(i,:))*(log(theta_class0_beta(a,:)))') + ((r_bin_dat(i,:))*(log(1-theta_class0_beta(a,:)))');
        %pzero(i) = exp(pzero(i));
        if pone(i)>pzero(i)
            r_test_label(i) = 1;
        else
            r_test_label(i) = 0;
        end
    end
    train_error(a) = 100*(1-(sum(not(xor(r_test_label,r_data_label)))/size(r_data_label,1)));
    if mod(a,20)==0
        display(['train iteration alpha #',num2str(a),' error = ',num2str(train_error(a)),'%']);
    end
end

%------------------------------------------------------------------------
% testing on testdata and calculating test error rate
%------------------------------------------------------------------------

t_test_label = 0*t_data_label;
pone = zeros(1,length(t_test_label));
pzero = zeros(1,length(t_test_label));
test_error = zeros(1,length(alpha));

for a=1:length(alpha)
    for i=1:length(t_test_label)
        pone(i) = log(prior_pi_one) + (t_bin_dat(i,:)*(log(theta_class1_beta(a,:)))') + ((1-t_bin_dat(i,:))*(log(1-theta_class1_beta(a,:)))');
        %pone(i) = exp(pone(i));
        pzero(i) = log(prior_pi_zero) + ((1-t_bin_dat(i,:))*(log(theta_class0_beta(a,:)))') + ((t_bin_dat(i,:))*(log(1-theta_class0_beta(a,:)))');
        %pzero(i) = exp(pzero(i));
        if pone(i)>pzero(i)
            t_test_label(i) = 1;
        else
            t_test_label(i) = 0;
        end
    end
    test_error(a) = 100*(1-(sum(not(xor(t_test_label,t_data_label)))/size(t_data_label,1)));
    
    if mod(a,20)==0
        display(['test iteration alpha #',num2str(a),' error = ',num2str(test_error(a)),'%']);
    end
end

%------------------------------------------------------------------------
% plotting routine
%------------------------------------------------------------------------
RGB = [0.9047 0.1918 0.1988;0.2941 0.5447 0.7494;0.3718 0.7176 0.3612;1.0000 0.5482 0.1000;0.8650 0.8110 0.4330;0.6859 0.4035 0.2412];
figure;
hold on;
plot(alpha,train_error,':o','Color',RGB(1,:),'LineWidth',1,'MarkerSize',3);
plot(alpha,test_error,':o','Color',RGB(3,:),'LineWidth',1,'MarkerSize',3);
legend('training error','test error');
title(['alpha vs classification error']);
xlabel('alpha');
ylabel('classification error %');
grid;
hold off;
display(['@alpha(1,10,100), training error rate = (',num2str(train_error(3)),'%, ', num2str(train_error(21)),'%, ',num2str(train_error(201)),'%)']);
display(['@alpha(1,10,100), test error rate = (',num2str(test_error(3)),'%, ', num2str(test_error(21)),'%, ',num2str(test_error(201)),'%)']);
