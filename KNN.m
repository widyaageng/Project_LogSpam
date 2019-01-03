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
t_data_label =spamData.ytest;
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

%----------------------------------------------------------------------
% Initiating neighbour count K and tracking vector for error calculation
%----------------------------------------------------------------------

% neighbour count
k_fix = [(1:1:9)';(10:5:100)'];

% training classification output tracking for each lambda
y_gauss_train = zeros(length(k_fix),length(r_data_label));
y_log_train = zeros(length(k_fix),length(r_data_label));
y_bin_train = zeros(length(k_fix),length(r_data_label));

% trainig classification error tracking for each lambda
gauss_error_train = zeros(1,length(k_fix));
log_error_train = zeros(1,length(k_fix));
bin_error_train = zeros(1,length(k_fix));

% training classification output tracking for each lambda
y_gauss_test = zeros(length(k_fix),length(t_data_label));
y_log_test = zeros(length(k_fix),length(t_data_label));
y_bin_test = zeros(length(k_fix),length(t_data_label));

% trainig classification error tracking for each lambda
gauss_error_test = zeros(1,length(k_fix));
log_error_test = zeros(1,length(k_fix));
bin_error_test = zeros(1,length(k_fix));

%------------------------------------------------------------------------------------
% Making distance database from one training data point to another training data point
%------------------------------------------------------------------------------------

dist_gauss_train = cell(length(r_data_label),1);
dist_log_train = cell(length(r_data_label),1);
dist_bin_train = cell(length(r_data_label),1);

% for progress bar
index = 0;
h = waitbar(0,'Please wait...');

for i=1:length(r_data_label)
    for j=1:(length(r_data_label))
        
        % skipped distances/non-calculated distances are set to large value
        % for easy searching of k shortest distances, in this case is the
        % self distance otherwise the minimal distance would be always zero
        % (self distance)
        
        if i==j
            dist_gauss_train{i}(j) = 1e3;
            dist_log_train{i}(j) = 1e3;
            dist_bin_train{i}(j) = 1e3;
        else
            % calculating distances, distance i to j
            dist_gauss_train{i}(j) = sqrt((r_norm_dat_gauss(i,:) - r_norm_dat_gauss(j,:))*(r_norm_dat_gauss(i,:) - r_norm_dat_gauss(j,:))');
            dist_log_train{i}(j) = sqrt((r_norm_dat_log(i,:) - r_norm_dat_log(j,:))*(r_norm_dat_log(i,:) - r_norm_dat_log(j,:))');
            dist_bin_train{i}(j) = sum(xor(r_bin_dat(i,:),r_bin_dat(j,:)));
            index = index + 1;
        end
    end
    
    waitbar(index/(length(r_data_label)*length(r_data_label)),h,sprintf('Generating distance database between training data...%2.1f%%',100*index/(length(r_data_label)*length(r_data_label))));
end

close(h);

%------------------------------------------------------------------------------------
% Making distance database for one test data point to another training data point
%------------------------------------------------------------------------------------

dist_gauss_test = cell(length(t_data_label),1);
dist_log_test= cell(length(t_data_label),1);
dist_bin_test = cell(length(t_data_label),1);

% for progress bar
index = 1;
h = waitbar(0,'Please wait...');

for i=1:length(t_data_label)
    for j=1:(length(r_data_label))
        
        % calculating distances from a test data to whole set of training data
        dist_gauss_test{i}(j) = sqrt((t_norm_dat_gauss(i,:) - r_norm_dat_gauss(j,:))*(t_norm_dat_gauss(i,:) - r_norm_dat_gauss(j,:))');
        dist_log_test{i}(j) = sqrt((t_norm_dat_log(i,:) - r_norm_dat_log(j,:))*(t_norm_dat_log(i,:) - r_norm_dat_log(j,:))');
        dist_bin_test{i}(j) = sum(xor(t_bin_dat(i,:),r_bin_dat(j,:)));
        index = index + 1;
    end
    
    waitbar(index/(length(t_data_label)*length(r_data_label)),h,sprintf('Getting distance database from test data to train data...%2.1f%%',100*index/(length(t_data_label)*length(r_data_label))));
end

close(h);

%--------------------------------------------------------------
% k iteration to check neighbour fixed count,K, effect on error
%--------------------------------------------------------------

for k=1:length(k_fix)
    disp('----------------------------------------');
    disp(['--------iteration for k = ',num2str(k_fix(k)),'--------']);

    
    %-------------------------------------------------
    % Labelling training data with k-nearest neighbour
    %-------------------------------------------------
    
    % gaussian data finding k-nearest neighbour
    for i=1:length(r_data_label)
        k_nearest_gauss = sort(dist_gauss_train{i}');
        k_nearest_gauss = find(dist_gauss_train{i}<=k_nearest_gauss(k_fix(k)));
        class1_prob = sum(and(ones(length(k_nearest_gauss),1),r_data_label(k_nearest_gauss')))/length(k_nearest_gauss);
        class0_prob = sum(not(or(zeros(length(k_nearest_gauss),1),r_data_label(k_nearest_gauss'))))/length(k_nearest_gauss);
        y_gauss_train(k,i) = 0;
        if class1_prob > class0_prob
            y_gauss_train(k,i) = 1;
        elseif class1_prob == class0_prob
            if rand(1) > 0.5
                y_gauss_train(k,i) = 1;
            else
                y_gauss_train(k,i) = 0;
            end
        end
    end
    
    % log data finding k-nearest neighbour
    for i=1:length(r_data_label)
        k_nearest_log = sort(dist_log_train{i}');
        k_nearest_log = find(dist_log_train{i}<=k_nearest_log(k_fix(k)));
        class1_prob = sum(and(ones(length(k_nearest_log),1),r_data_label(k_nearest_log')))/length(k_nearest_log);
        class0_prob = sum(not(or(zeros(length(k_nearest_log),1),r_data_label(k_nearest_log'))))/length(k_nearest_log);
        y_log_train(k,i) = 0;
        if class1_prob > class0_prob
            y_log_train(k,i) = 1;
        elseif class1_prob == class0_prob
            if rand(1) > 0.5
                y_log_train(k,i) = 1;
            else
                y_log_train(k,i) = 0;
            end
        end
    end
    
    % bin data finding k-nearest neighbour
    for i=1:length(r_data_label)
        k_nearest_bin = sort(dist_bin_train{i}');
        k_nearest_bin = find(dist_bin_train{i}<=k_nearest_bin(k_fix(k)));
        class1_prob = sum(and(ones(length(k_nearest_bin),1),r_data_label(k_nearest_bin')))/length(k_nearest_bin);
        class0_prob = sum(not(or(zeros(length(k_nearest_bin),1),r_data_label(k_nearest_bin'))))/length(k_nearest_bin);
        y_bin_train(k,i) = 0;
        if class1_prob > class0_prob
            y_bin_train(k,i) = 1;
        elseif class1_prob == class0_prob
            if rand(1) > 0.5
                y_bin_train(k,i) = 1;
            else
                y_bin_train(k,i) = 0;
            end
        end
    end
    
    % training data classification error calculation
    gauss_error_train(k) = 100*sum(xor(y_gauss_train(k,:)', r_data_label))/length(r_data_label);
    log_error_train(k) = 100*sum(xor(y_log_train(k,:)', r_data_label))/length(r_data_label);
    bin_error_train(k) = 100*sum(xor(y_bin_train(k,:)', r_data_label))/length(r_data_label);
    
    disp(['Classification error in gaussian training data = ', num2str(gauss_error_train(k)),'%']);
    disp(['Classification error in log training  data = ', num2str(log_error_train(k)),'%']);
    disp(['Classification error in binary training data = ', num2str(bin_error_train(k)),'%']);
    
    %---------------------------------------------------------------------
    % Labelling test data with k-nearest neighbour label map from original
    % label of training data
    %---------------------------------------------------------------------
    
    % gaussian data, testdata labelling
    for i=1:length(t_data_label)
        k_nearest_gauss = sort(dist_gauss_test{i}');
        k_nearest_gauss = find(dist_gauss_test{i}<=k_nearest_gauss(k_fix(k)));
        class1_prob = sum(and(ones(length(k_nearest_gauss),1),r_data_label(k_nearest_gauss')))/length(k_nearest_gauss);
        class0_prob = sum(not(or(zeros(length(k_nearest_gauss),1),r_data_label(k_nearest_gauss'))))/length(k_nearest_gauss);
        y_gauss_test(k,i) = 0;
        if class1_prob > class0_prob
            y_gauss_test(k,i) = 1;
        elseif class1_prob == class0_prob
            if rand(1) > 0.5
                y_gauss_test(k,i) = 1;
            else
                y_gauss_test(k,i) = 0;
            end
        end
    end
    
    % log data, testdata labelling
    for i=1:length(t_data_label)
        k_nearest_log = sort(dist_log_test{i}');
        k_nearest_log = find(dist_log_test{i}<=k_nearest_log(k_fix(k)));
        class1_prob = sum(and(ones(length(k_nearest_log),1),r_data_label(k_nearest_log')))/length(k_nearest_log);
        class0_prob = sum(not(or(zeros(length(k_nearest_log),1),r_data_label(k_nearest_log'))))/length(k_nearest_log);
        y_log_test(k,i) = 0;
        if class1_prob > class0_prob
            y_log_test(k,i) = 1;
        elseif class1_prob == class0_prob
            if rand(1) > 0.5
                y_log_test(k,i) = 1;
            else
                y_log_test(k,i) = 0;
            end
        end
    end
    
    % bin data, testdata labelling
    for i=1:length(t_data_label)
        k_nearest_bin = sort(dist_bin_test{i}');
        k_nearest_bin = find(dist_bin_test{i}<=k_nearest_bin(k_fix(k)));
        class1_prob = sum(and(ones(length(k_nearest_bin),1),r_data_label(k_nearest_bin')))/length(k_nearest_bin);
        class0_prob = sum(not(or(zeros(length(k_nearest_bin),1),r_data_label(k_nearest_bin'))))/length(k_nearest_bin);
        y_bin_tset(k,i) = 0;
        if class1_prob > class0_prob
            y_bin_test(k,i) = 1;
        elseif class1_prob == class0_prob
            if rand(1) > 0.5
                y_bin_test(k,i) = 1;
            else
                y_bin_test(k,i) = 0;
            end
        end
    end
    
    % training data classification error calculation
    gauss_error_test(k) = 100*sum(xor(y_gauss_test(k,:)', t_data_label))/length(t_data_label);
    log_error_test(k) = 100*sum(xor(y_log_test(k,:)', t_data_label))/length(t_data_label);
    bin_error_test(k) = 100*sum(xor(y_bin_test(k,:)', t_data_label))/length(t_data_label);
    
    disp(['Classification error in gaussian test data = ', num2str(gauss_error_test(k)),'%']);
    disp(['Classification error in log test  data = ', num2str(log_error_test(k)),'%']);
    disp(['Classification error in binary test data = ', num2str(bin_error_test(k)),'%']);
end

disp(['@K(1,10,100), training error rate (gaussian) = (',num2str(gauss_error_train(1)),'%, ', num2str(gauss_error_train(10)),'%, ',num2str(gauss_error_train(28)),'%)']);
disp(['@K(1,10,100), training error rate (log) = (',num2str(log_error_train(1)),'%, ', num2str(log_error_train(10)),'%, ',num2str(log_error_train(28)),'%)']);
disp(['@K(1,10,100), training error rate (bin) = (',num2str(bin_error_train(1)),'%, ', num2str(bin_error_train(10)),'%, ',num2str(bin_error_train(28)),'%)']);

disp(['@K(1,10,100), test error rate (gaussian) = (',num2str(gauss_error_test(1)),'%, ', num2str(gauss_error_test(10)),'%, ',num2str(gauss_error_test(28)),'%)']);
disp(['@K(1,10,100), test error rate (log) = (',num2str(log_error_test(1)),'%, ', num2str(log_error_test(10)),'%, ',num2str(log_error_test(28)),'%)']);
disp(['@K(1,10,100), test error rate (bin) = (',num2str(bin_error_test(1)),'%, ', num2str(bin_error_test(10)),'%, ',num2str(bin_error_test(28)),'%)']);

%------------------------------------------------------------------------
% plotting routine
%------------------------------------------------------------------------
RGB = [0.9047 0.1918 0.1988;0.2941 0.5447 0.7494;0.3718 0.7176 0.3612;1.0000 0.5482 0.1000;0.8650 0.8110 0.4330;0.6859 0.4035 0.2412];
figure;
hold on;
plot(k_fix,gauss_error_train,'-x','Color',RGB(1,:),'LineWidth',1,'MarkerSize',7);
plot(k_fix,log_error_train,'-x','Color',RGB(2,:),'LineWidth',1,'MarkerSize',7);
plot(k_fix,bin_error_train,'-x','Color',RGB(3,:),'LineWidth',1,'MarkerSize',7);
plot(k_fix,gauss_error_test,'-o','Color',RGB(1,:),'LineWidth',1,'MarkerSize',7);
plot(k_fix,log_error_test,'-o','Color',RGB(2,:),'LineWidth',1,'MarkerSize',7);
plot(k_fix,bin_error_test,'-o','Color',RGB(3,:),'LineWidth',1,'MarkerSize',7);
legend('gauss training error','log training error','bin training error','gauss test error','log test error','bin test error');
title('neighbour sample (K) vs classification error');
xlabel('K neighbour');
ylabel('classification error %');
grid on;
hold off;