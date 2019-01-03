clear;clc;
spamData = load('spamData.mat');

%traindata
r_data_feat = spamData.Xtrain;
r_data_label = spamData.ytrain;
r_norm_dat_gauss = zeros(size(r_data_feat,1),size(r_data_feat,2));
r_norm_dat_log = zeros(size(r_data_feat,1),size(r_data_feat,2));
r_bin_dat = zeros(size(r_data_feat,1),size(r_data_feat,2));
N = size(r_data_feat,1);


%testdata
t_data_feat = spamData.Xtest;
t_data_label = spamData.ytest;
t_norm_dat_gauss = zeros(size(t_data_feat,1),size(t_data_feat,2));
t_norm_dat_log = zeros(size(t_data_feat,1),size(t_data_feat,2));
t_bin_dat = zeros(size(t_data_feat,1),size(t_data_feat,2));

%-------------------------------------------------
% Data Preprocessing
%-------------------------------------------------

%--Feature Normalization : gaussian , log offset --

%traindata normalization using gaussian and log
for i=1:size(r_data_feat,2)
    r_norm_dat_gauss(:,i) = zscore(r_data_feat(:,i));
    r_norm_dat_log(:,i) = log(r_data_feat(:,i) + 0.1);
end


%testdata normalization with gaussian and log
test_offset = zeros(1,size(t_data_feat,2));
for i=1:size(t_data_feat,2)
    t_norm_dat_gauss(:,i) = zscore(t_data_feat(:,i));
    t_norm_dat_log(:,i) = log(t_data_feat(:,i) + 0.1);
end

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

%---------------------------------------------------------
% data concatenation and regularization constant generation
%---------------------------------------------------------

% regularization constant
lambda = [(1:1:9)';(10:5:100)'];

% training classification output tracking for each lambda
y_gauss_train = zeros(length(lambda),length(r_data_label));
y_log_train = zeros(length(lambda),length(r_data_label));
y_bin_train = zeros(length(lambda),length(r_data_label));

% trainig classification error tracking for each lambda
gauss_error_train = zeros(1,length(lambda));
log_error_train = zeros(1,length(lambda));
bin_error_train = zeros(1,length(lambda));

% test classification output tracking for each lambda
y_gauss_test = zeros(length(lambda),length(t_data_label));
y_log_test = zeros(length(lambda),length(t_data_label));
y_bin_test = zeros(length(lambda),length(t_data_label));

% test classification error tracking for each lambda
gauss_error_test = zeros(1,length(lambda));
log_error_test = zeros(1,length(lambda));
bin_error_test = zeros(1,length(lambda));

% concatenated training data for each version of data, column will be data
% point and row will be feature number, ie : trainX_dat_gauss(i,j) contains
% feature i value in training data j
trainX_dat_gauss = r_norm_dat_gauss';
trainX_dat_log = r_norm_dat_log';
trainX_dat_bin = r_bin_dat';

% concatenated test data for each version of data
testX_dat_gauss = t_norm_dat_gauss';
testX_dat_log = t_norm_dat_log';
testX_dat_bin = t_bin_dat';

%---------------------------------------------------------
% lambda iteration to check regularization effect on error
%---------------------------------------------------------

% quadratic difference of stopping criteria
stopping_crit = 1e-5;

for k=1:length(lambda)
    display(['----------------------------------------']);
    display(['--------iteration for lambda = ',num2str(lambda(k)),'--------']);
    % weight for gaussian data
    weight_prev_gauss = 1e5*ones(size(r_data_feat,2) + 1,1);
    weight_now_gauss = zeros(size(r_data_feat,2) + 1,1);
    diff_weight_gauss = weight_now_gauss - weight_prev_gauss;
    
    % weight for log data
    weight_prev_log = 1e5*ones(size(r_data_feat,2) + 1,1);
    weight_now_log = zeros(size(r_data_feat,2) + 1,1);
    diff_weight_log = weight_now_log - weight_prev_log;
    
    % weight for binary data
    weight_prev_bin = 1e5*ones(size(r_data_feat,2) + 1,1);
    weight_now_bin = zeros(size(r_data_feat,2) + 1,1);
    diff_weight_bin = weight_now_bin - weight_prev_bin;
    
    % logistic value for each version of data, initial set at 0.5 (middle)
    mu_gauss = ones(length(r_data_label),1)*0.5;
    mu_log = ones(length(r_data_label),1)*0.5;
    mu_bin = ones(length(r_data_label),1)*0.5;
    
    % iteration tracker
    gauss_iteration_num = 0;
    log_iteration_num = 0;
    bin_iteration_num = 0;
    
    %-------------------------------------------------
    % Weight iteration using training data
    %-------------------------------------------------
    
    % optimizing weight on gaussian training data, stop when stop crit is
    % met against square of weight vector increment
    while (sum(abs(diff_weight_gauss)) > stopping_crit)
        gauss_iteration_num = gauss_iteration_num + 1;
        mu_gauss = ((1+exp((-1)*weight_now_gauss'*([ones(1,length(r_data_label));trainX_dat_gauss]))).^(-1));
        
        % hessian rate for gaussian data
        hessian_gauss = ([ones(1,length(r_data_label));trainX_dat_gauss])*diag(mu_gauss.*(1-mu_gauss))*([ones(1,length(r_data_label));trainX_dat_gauss])';
        hessian_gauss = hessian_gauss + lambda(k)*diag([0 ones(1,size(r_data_feat,2))]);
        
        if mod(gauss_iteration_num,100)==0
            display(['gauss iteration #',num2str(gauss_iteration_num)]);
        end
        
        % regularized gradient calculation on gaussian data, ignoring bias weight
        gradient_gauss = ([ones(1,length(r_data_label));trainX_dat_gauss])*(mu_gauss' - r_data_label) + ([0;(lambda(k)*ones(size(r_data_feat,2),1))].*weight_now_gauss);
        weight_prev_gauss = weight_now_gauss;
        weight_now_gauss = weight_now_gauss - (hessian_gauss\gradient_gauss);
        diff_weight_gauss = weight_now_gauss - weight_prev_gauss;
    end
    
    % optimizing weight on log training data, stop when stop crit is
    % met against square of weight vector increment
    while (sum(abs(diff_weight_log)) > stopping_crit)
        log_iteration_num = log_iteration_num + 1;
        mu_log = ((1+exp((-1)*weight_now_log'*([ones(1,length(r_data_label));trainX_dat_log]))).^(-1));
        
        % hessian rate for log data
        hessian_log = ([ones(1,length(r_data_label));trainX_dat_log])*diag(mu_log.*(1-mu_log))*([ones(1,length(r_data_label));trainX_dat_log])';
        hessian_log = hessian_log + lambda(k)*diag([0 ones(1,size(r_data_feat,2))]);
        
        if mod(log_iteration_num,100)==0
            display(['log iteration #',num2str(log_iteration_num)]);
        end
        
        % regularized gradient calculation on log data, ignoring bias weight
        gradient_log = ([ones(1,length(r_data_label));trainX_dat_log])*(mu_log' - r_data_label) + ([0;(lambda(k)*ones(size(r_data_feat,2),1))].*weight_now_log);
        weight_prev_log = weight_now_log;
        weight_now_log = weight_now_log - (hessian_log\gradient_log);
        diff_weight_log = weight_now_log - weight_prev_log;
    end
    
    % optimizing weight on binary training data, stop when stop crit is
    % met against square of weight vector increment
    while (sum(abs(diff_weight_bin)) > stopping_crit)
        bin_iteration_num = bin_iteration_num + 1;
        mu_bin = ((1+exp((-1)*weight_now_bin'*([ones(1,length(r_data_label));trainX_dat_bin]))).^(-1));
        
        % hessian rate  for binary data
        hessian_bin = ([ones(1,length(r_data_label));trainX_dat_bin])*diag(mu_bin.*(1-mu_bin))*([ones(1,length(r_data_label));trainX_dat_bin])';
        hessian_bin = hessian_bin + lambda(k)*diag([0 ones(1,size(r_data_feat,2))]);
    
        if mod(bin_iteration_num,100)==0
            display(['binary data iteration #',num2str(bin_iteration_num)]);
        end
        
        % regularized gradient calculation on binary data, ignoring bias weight
        gradient_bin = ([ones(1,length(r_data_label));trainX_dat_bin])*(mu_bin' - r_data_label) + ([0;(lambda(k)*ones(size(r_data_feat,2),1))].*weight_now_bin);
        weight_prev_bin = weight_now_bin;
        weight_now_bin = weight_now_bin - (hessian_bin\gradient_bin);
        diff_weight_bin = weight_now_bin - weight_prev_bin;
    end
    
    %-------------------------------------------------
    % Test the obtained weight with training data
    %-------------------------------------------------
    
    % sigmoid value calculation for each type of training data for class 1 (spam) probability
    y_gauss_train(k,:) = ((1+exp((-1)*weight_now_gauss'*([ones(1,length(r_data_label));trainX_dat_gauss]))).^(-1));
    y_log_train(k,:) = ((1+exp((-1)*weight_now_log'*([ones(1,length(r_data_label));trainX_dat_log]))).^(-1));
    y_bin_train(k,:) = ((1+exp((-1)*weight_now_bin'*([ones(1,length(r_data_label));trainX_dat_bin]))).^(-1));
    
    % training data classification based on class 1 (spam) probability, and error calculation
    % if class 1 probability is more than or equal to 0.5, then assign and compute mismatch using sum(xor(..,..))
    gauss_error_train(k) = 100*sum(xor((y_gauss_train(k,:) > 0.5)', r_data_label))/length(r_data_label);
    log_error_train(k) = 100*sum(xor((y_log_train(k,:) > 0.5)', r_data_label))/length(r_data_label);
    bin_error_train(k) = 100*sum(xor((y_bin_train(k,:) > 0.5)', r_data_label))/length(r_data_label);
    
    display(['Classification error in gaussian training data = ', num2str(gauss_error_train(k)),'%']);
    display(['Classification error in log training  data = ', num2str(log_error_train(k)),'%']);
    display(['Classification error in binary training data = ', num2str(bin_error_train(k)),'%']);
    
    %-------------------------------------------------
    % Test the obtained weight with test data
    %-------------------------------------------------
    
    % sigmoid value calculation for each type of test data
    y_gauss_test(k,:) = ((1+exp((-1)*weight_now_gauss'*([ones(1,length(t_data_label));testX_dat_gauss]))).^(-1));
    y_log_test(k,:) = ((1+exp((-1)*weight_now_log'*([ones(1,length(t_data_label));testX_dat_log]))).^(-1));
    y_bin_test(k,:) = ((1+exp((-1)*weight_now_bin'*([ones(1,length(t_data_label));testX_dat_bin]))).^(-1));
    
    % test data classification based on sigmoid value (threshold = 0.5), and error calculation
    gauss_error_test(k) = 100*sum(xor((y_gauss_test(k,:) > 0.5)', t_data_label))/length(t_data_label);
    log_error_test(k) = 100*sum(xor((y_log_test(k,:) > 0.5)', t_data_label))/length(t_data_label);
    bin_error_test(k) = 100*sum(xor((y_bin_test(k,:) > 0.5)', t_data_label))/length(t_data_label);
    
    display(['Classification error in gaussian test data = ', num2str(gauss_error_test(k)),'%']);
    display(['Classification error in log test data = ', num2str(log_error_test(k)),'%']);
    display(['Classification error in binary test data = ', num2str(bin_error_test(k)),'%']);
end

%------------------------------------------------------------------------
% plotting routine
%------------------------------------------------------------------------
RGB = [0.9047 0.1918 0.1988;0.2941 0.5447 0.7494;0.3718 0.7176 0.3612;1.0000 0.5482 0.1000;0.8650 0.8110 0.4330;0.6859 0.4035 0.2412];
figure;
hold on;
plot(lambda,gauss_error_train,'-x','Color',RGB(1,:),'LineWidth',1,'MarkerSize',7);
plot(lambda,log_error_train,'-x','Color',RGB(2,:),'LineWidth',1,'MarkerSize',7);
plot(lambda,bin_error_train,'-x','Color',RGB(3,:),'LineWidth',1,'MarkerSize',7);
plot(lambda,gauss_error_test,'-o','Color',RGB(1,:),'LineWidth',1,'MarkerSize',7);
plot(lambda,log_error_test,'-o','Color',RGB(2,:),'LineWidth',1,'MarkerSize',7);
plot(lambda,bin_error_test,'-o','Color',RGB(3,:),'LineWidth',1,'MarkerSize',7);
legend('gauss training error','log training error','bin training error','gauss test error','log test error','bin test error');
title('regularization vs classification error');
xlabel('lambda');
ylabel('classification error %');
grid;
hold off;

display(['@lambda(1,10,100), training error rate (gaussian) = (',num2str(gauss_error_train(1)),'%, ', num2str(gauss_error_train(10)),'%, ',num2str(gauss_error_train(28)),'%)']);
display(['@lambda(1,10,100), training error rate (log) = (',num2str(log_error_train(1)),'%, ', num2str(log_error_train(10)),'%, ',num2str(log_error_train(28)),'%)']);
display(['@lambda(1,10,100), training error rate (bin) = (',num2str(bin_error_train(1)),'%, ', num2str(bin_error_train(10)),'%, ',num2str(bin_error_train(28)),'%)']);

display(['@lambda(1,10,100), test error rate (gaussian) = (',num2str(gauss_error_test(1)),'%, ', num2str(gauss_error_test(10)),'%, ',num2str(gauss_error_test(28)),'%)']);
display(['@lambda(1,10,100), test error rate (log) = (',num2str(log_error_test(1)),'%, ', num2str(log_error_test(10)),'%, ',num2str(log_error_test(28)),'%)']);
display(['@lambda(1,10,100), test error rate (bin) = (',num2str(bin_error_test(1)),'%, ', num2str(bin_error_test(10)),'%, ',num2str(bin_error_test(28)),'%)']);



















