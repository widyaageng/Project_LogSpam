clear;clc;
load('Database.mat');

%--------------------------------------------------------------
% k iteration to check neighbour fixed count,K, effect on error
%--------------------------------------------------------------

for k=1:length(k_fix)
    display('----------------------------------------');
    display(['--------iteration for k = ',num2str(k_fix(k)),'--------']);

    
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
    
    display(['Classification error in gaussian training data = ', num2str(gauss_error_train(k)),'%']);
    display(['Classification error in log training  data = ', num2str(log_error_train(k)),'%']);
    display(['Classification error in binary training data = ', num2str(bin_error_train(k)),'%']);
    
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
    
    display(['Classification error in gaussian test data = ', num2str(gauss_error_test(k)),'%']);
    display(['Classification error in log test  data = ', num2str(log_error_test(k)),'%']);
    display(['Classification error in binary test data = ', num2str(bin_error_test(k)),'%']);
end

display(['@K(1,10,100), training error rate (gaussian) = (',num2str(gauss_error_train(1)),'%, ', num2str(gauss_error_train(10)),'%, ',num2str(gauss_error_train(28)),'%)']);
display(['@K(1,10,100), training error rate (log) = (',num2str(log_error_train(1)),'%, ', num2str(log_error_train(10)),'%, ',num2str(log_error_train(28)),'%)']);
display(['@K(1,10,100), training error rate (bin) = (',num2str(bin_error_train(1)),'%, ', num2str(bin_error_train(10)),'%, ',num2str(bin_error_train(28)),'%)']);

display(['@K(1,10,100), test error rate (gaussian) = (',num2str(gauss_error_test(1)),'%, ', num2str(gauss_error_test(10)),'%, ',num2str(gauss_error_test(28)),'%)']);
display(['@K(1,10,100), test error rate (log) = (',num2str(log_error_test(1)),'%, ', num2str(log_error_test(10)),'%, ',num2str(log_error_test(28)),'%)']);
display(['@K(1,10,100), test error rate (bin) = (',num2str(bin_error_test(1)),'%, ', num2str(bin_error_test(10)),'%, ',num2str(bin_error_test(28)),'%)']);


 
%------------------------------------------------------------------------
% plotting routine
%------------------------------------------------------------------------
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