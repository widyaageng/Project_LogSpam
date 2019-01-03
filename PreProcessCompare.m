clear;clc;
spamData = load('spamData.mat');

%traindata
r_data_feat = spamData.Xtrain;
r_data_label = spamData.ytrain;
r_norm_dat_gauss = zeros(size(r_data_feat,1),size(r_data_feat,2));
r_norm_dat_log = zeros(size(r_data_feat,1),size(r_data_feat,2));
N = size(r_data_feat,1);


%testdata
t_data_feat = spamData.Xtest;
t_data_label =spamData.ytest;
t_norm_dat_gauss = zeros(size(t_data_feat,1),size(t_data_feat,2));
t_norm_dat_log = zeros(size(t_data_feat,1),size(t_data_feat,2));

%-------------------------------------------------
%preprocessing
%-------------------------------------------------

%--Feature Normalization : gaussian , log offset --

%traindata normalization using gaussian and log
for i=1:size(r_data_feat,2)
    r_norm_dat_gauss(:,i) = zscore(r_data_feat(:,i));
    r_norm_dat_log(:,i) = log(r_data_feat(:,i) + 0.1);
end


%testdata normalization with log and gaussian
test_offset = zeros(1,size(t_data_feat,2));
for i=1:size(t_data_feat,2)
    t_norm_dat_gauss(:,i) = zscore(t_data_feat(:,i));
    t_norm_dat_log(:,i) = log(t_data_feat(:,i) + 0.1);
end

%-------------------------------------------------
% estimating prior probability of class/label using MLE
%-------------------------------------------------

% # of data from r_data_feat that belongs to class 1 (spam)
Nclass_1 = 0;

%create a matrix whose rows are data vector that is classed as '1'
class1_data_gauss = [];
class1_data_log = [];

%create a matrix whose rows are data vector that is classed as '0'
class0_data_gauss = [];
class0_data_log = [];

% separating data into each class, 1 and 0, and in each datatype (gauss/log)
for i=1:size(r_data_feat,1)
    if r_data_label(i)>0.5 %to prevent floating issue at 'near' 0 or 1
        Nclass_1 = Nclass_1 + 1;
        class1_data_gauss = [class1_data_gauss;r_norm_dat_gauss(i,:)];
        class1_data_log = [class1_data_log;r_norm_dat_log(i,:)];
    else
        class0_data_gauss = [class0_data_gauss;r_norm_dat_gauss(i,:)];
        class0_data_log = [class0_data_log;r_norm_dat_log(i,:)];
    end
end

%MLE prior probability of class/label
prior_pi_one = Nclass_1/size(r_data_feat,1);
prior_pi_zero = 1 - prior_pi_one;

% ML Estimation of mean (mu) and variance (ohm)
% mu_matrix is matrix whose element mu(i,j) is the average value of feature j-th in class (i-1)-th
% gauss suffix corresponds to gaussian data
mu_matrix_gauss = [mean(class0_data_gauss);mean(class1_data_gauss)];
% log suffix corresponds to log data
mu_matrix_log = [mean(class0_data_log);mean(class1_data_log)];

% ohm_matrix is matrix whose element ohm(i-j) is the variance (std^2) value of feature j-th in class (i-1)-th
ohm_matrix_gauss = ([std(class0_data_gauss);std(class1_data_gauss)]).^2;
ohm_matrix_log = ([std(class0_data_log);std(class1_data_log)]).^2;

%------------------------------------------------------------------------
% testing on training data/validation and calculating training error rate
%------------------------------------------------------------------------
% initiating label for testing with train data, gaussian and log
r_test_label_log = 0*r_data_label;
r_test_label_gauss = 0*r_data_label;

% initiating class probability given data x, p(y=c|x), where c =0,1
pone_train_gauss = zeros(1,length(r_test_label_gauss)); %for class 1, with gaussian data
pzero_train_gauss = zeros(1,length(r_test_label_gauss)); %for class 0, with gaussian data
pone_train_log = zeros(1,length(r_test_label_log)); %for class 1, with log data
pzero_train_log = zeros(1,length(r_test_label_log)); %for class 0, with log data

% training error, index 1 is for gaussian feature, and index 2 is for log feature
train_error = zeros(1,2);

for i=1:length(r_data_label)
    % putting ML plugin of prior class probability, gaussian data
    pone_train_gauss(i) = log(prior_pi_one);
    pzero_train_gauss(i) = log(prior_pi_zero);
    
    % putting ML plugin of prior class probability, log data
    pone_train_log(i) = log(prior_pi_one);
    pzero_train_log(i) = log(prior_pi_zero);
    
    % iterate to each feature on each data row vector j
    for j=1:length(r_data_feat(i,:))
        % class proability using gaussian train data, log p(y|x) = log (piML) - 0.5 log (sqrt(2*pi*sig^2) - ((x-mu)^2/(2*sig^2)), iteration update
        pone_train_gauss(i) = pone_train_gauss(i) - (0.5*log(2*pi*ohm_matrix_gauss(2,j))) - (((r_norm_dat_gauss(i,j) - mu_matrix_gauss(2,j))^2)/(2*ohm_matrix_gauss(2,j)));
        pzero_train_gauss(i) = pzero_train_gauss(i) - (0.5*log(2*pi*ohm_matrix_gauss(1,j))) - (((r_norm_dat_gauss(i,j) - mu_matrix_gauss(1,j))^2)/(2*ohm_matrix_gauss(1,j)));
        
        % class proability using log train data, log p(y|x) = log (piML) - 0.5 log (sqrt(2*pi*sig^2) - ((x-mu)/(2*sig^2)), iteration update
        pone_train_log(i) = pone_train_log(i) - (0.5*log(2*pi*ohm_matrix_log(2,j))) - (((r_norm_dat_log(i,j) - mu_matrix_log(2,j))^2)/(2*ohm_matrix_log(2,j)));
        pzero_train_log(i) = pzero_train_log(i) - (0.5*log(2*pi*ohm_matrix_log(1,j))) - (((r_norm_dat_log(i,j) - mu_matrix_log(1,j))^2)/(2*ohm_matrix_log(1,j)));   
    end
    
    if pone_train_gauss(i)>pzero_train_gauss(i)
        r_test_label_gauss(i) = 1;
    else
        r_test_label_gauss(i) = 0;
    end
     
    if pone_train_log(i)>pzero_train_log(i)
        r_test_label_log(i) = 1;
    else
        r_test_label_log(i) = 0;
    end
    
    if or(mod(i,500)==0,mod(i,length(r_data_label))==0)
        display(['train iteration #',num2str(i),' out of ', num2str(length(r_data_label))]);
    end
end

%training error for gaussian data
train_error(1) = 100*((sum(xor(r_test_label_gauss,r_data_label))/size(r_data_label,1)));
disp('------------------------------------------------');
disp(['train error with gaussian data = ',num2str(train_error(1)),'%']);

%training error for log data
train_error(2) = 100*((sum(xor(r_test_label_log,r_data_label))/size(r_data_label,1)));
disp(['train error with log data = ',num2str(train_error(2)),'%']);
disp('------------------------------------------------');

%------------------------------------------------------------------------
% testing on testdata and calculating test error rate
%------------------------------------------------------------------------
% initiating gaussian test label with test data and class probability using gaussian data
t_test_label_gauss = 0*t_data_label;
pone_test_gauss = zeros(1,length(t_test_label_gauss));
pzero_test_gauss = zeros(1,length(t_test_label_gauss));

% initiating log test label with test data and class probability using log data
t_test_label_log = 0*t_data_label;
pone_test_log = zeros(1,length(t_test_label_log));
pzero_test_log = zeros(1,length(t_test_label_log));

%test error with testdata, index 1 is for gaussian feature, and index 2 is for log feature
test_error = zeros(1,2);

for i=1:length(t_test_label_gauss)
    % putting ML plugin of prior class probability, gaussian data
    pone_test_gauss(i) = log(prior_pi_one);
    pzero_test_gauss(i) = log(prior_pi_zero);
    
    % putting ML plugin of prior class probability, log data
    pone_test_log(i) = log(prior_pi_one);
    pzero_test_log(i) = log(prior_pi_zero);
    
    for j=1:length(r_data_feat(i,:))
        % class proability using gaussian test data, log p(y|x) = log (piML) - 0.5 log (sqrt(2*pi*sig^2) - ((x-mu)/(2*sig^2))
        pone_test_gauss(i) = pone_test_gauss(i) - (0.5*log(2*pi*ohm_matrix_gauss(2,j))) - (((t_norm_dat_gauss(i,j) - mu_matrix_gauss(2,j))^2)/(2*ohm_matrix_gauss(2,j)));
        pzero_test_gauss(i) = pzero_test_gauss(i) - (0.5*log(2*pi*ohm_matrix_gauss(1,j))) - (((t_norm_dat_gauss(i,j) - mu_matrix_gauss(1,j))^2)/(2*ohm_matrix_gauss(1,j)));
        
        % class probability using log test data, log p(y|x) = log (piML) - 0.5 log (sqrt(2*pi*sig^2) - ((x-mu)/(2*sig^2))
        pone_test_log(i) = pone_test_log(i) - (0.5*log(2*pi*ohm_matrix_log(2,j))) - (((t_norm_dat_log(i,j) - mu_matrix_log(2,j))^2)/(2*ohm_matrix_log(2,j)));
        pzero_test_log(i) = pzero_test_log(i) - (0.5*log(2*pi*ohm_matrix_log(1,j))) - (((t_norm_dat_log(i,j) - mu_matrix_log(1,j))^2)/(2*ohm_matrix_log(1,j)));
    end
    
    if pone_test_gauss(i)>pzero_test_gauss(i)
        t_test_label_gauss(i) = 1;
    else
        t_test_label_gauss(i) = 0;
    end
     
    if pone_test_log(i)>pzero_test_log(i)
        t_test_label_log(i) = 1;
    else
        t_test_label_log(i) = 0;
    end
    
    if or(mod(i,500)==0,mod(i,length(t_test_label_gauss))==0)
        display(['test iteration #',num2str(i),' out of ', num2str(length(t_data_label))]);
    end
end

% calculating error on test with gaussian test data (index 1) and test with log test data (index 2)
test_error(1) = 100*((sum(xor(t_test_label_gauss,t_data_label))/size(t_data_label,1)));
test_error(2) = 100*((sum(xor(t_test_label_log,t_data_label))/size(t_data_label,1)));

% displaying classification error
disp('------------------------------------------------');
disp(['test error with gaussian data = ',num2str(test_error(1)),'%']);
disp(['test error with log data = ',num2str(test_error(2)),'%']);
disp('------------------------------------------------');