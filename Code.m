clc; clear all;
%% Importing the data
data_raw = readtable('Data.xlsx');

%% Exploratory data analysis
% Check the name and type of the columns
data_raw.Properties
data_values = table2array(data_raw);

% Look for empty values in each column
disp('Number of empty value in each column')
sum(isnan(data_values))

% Plot matrix
figure('Name','Visualisation of features')
plotmatrix(data_values)

% Basic statistics
stats_mean = mean(data_values);
stats_median = median(data_values);
stats_max = max(data_values);
stats_mean = min(data_values);
stats_range = range(data_values);
stats_stdev = std(data_values);
stats_skew = skewness(data_values);

disp('Exploratory data analysis')
stats_value = [stats_mean;stats_median;stats_max;stats_mean;stats_range;stats_stdev;stats_skew];
var_AT = stats_value(:,1);
var_V = stats_value(:,2);
var_AP = stats_value(:,3);
var_RH = stats_value(:,4);
var_PE = stats_value(:,5);
    
stats_label = {'meanD';'medD';'maxD';'minD';'rangeD';'stdD';'skewD'};
stats = table(stats_label,var_AT,var_V,var_AP,var_RH,var_PE);
disp(stats)

% Correlation matrix
[stats_corr,stats_Pvalue] = corrcoef(data_values);
disp('Correlation matrix')
disp(stats_corr)
disp('P-value')
disp(stats_Pvalue)

%% Data preprocessing
% Normalising data
data_normalised = (data_values - mean(data_values))./std(data_values)

% Shuffling
rng(1);
data_normalised = data_normalised(randperm(end),:);

% Splitting the data into trainning set and test set
hold_out_ratio = 0.2;                                      %%%Customisation
[data_rows,data_columns] = size(data_values);
train_size = round(data_rows*(1-hold_out_ratio));
test_size = data_rows - train_size;

X_train = data_normalised(1:train_size,1:data_columns-1);
Y_train = data_normalised(1:train_size,end);
X_test = data_normalised(1:test_size,1:data_columns-1);
Y_test = data_normalised(1:test_size,end);

%% Model fitting - Linear regression
Model_LR = fitlm(X_train, Y_train,'RobustOpts', 'off')
Model_LR_Y_pred = predict(Model_LR,X_test);

%% Model fitting - Random forest regression
Model_RF_MaxTrees = 100;                                   %%%Customisation
Model_RF_NumPredSample = 3;                                %%%Customisation
Model_RF_RMSE_test = zeros(Model_RF_MaxTrees,Model_RF_NumPredSample);

rng(2) % For reproducibility
for i = 1:Model_RF_NumPredSample
    for k = 1:Model_RF_MaxTrees
        Model_RF = TreeBagger(k,X_train,Y_train,'Method','Regression','NumPredictorsToSample',i);
        Model_RF_Y_pred = predict(Model_RF,X_test);
        Model_RF_RMSE_test(k,i) = sqrt(mean((Y_test - Model_RF_Y_pred).^2));
    end
end
%% Model evaluation - Linear Regression
% Calculating RMSE in test set
Model_LR_Residual_test = (Y_test - Model_LR_Y_pred).^2
Model_LR_RMSE_test = sqrt(mean((Y_test - Model_LR_Y_pred).^2));

% Plotting residuals
figure('Name','Model residual')
scatter(Model_LR_Residual_test, Model_LR_Y_pred)
xlabel('Residual')
ylabel('Predicted')

% Removing outliers
[Model_LR_Residual_test_sorted,IX] = sort(Model_LR_Residual_test,'descend');
Model_LR_Y_pred_sorted = Model_LR_Y_pred(IX,:);
Resplot = [Model_LR_Residual_test_sorted Model_LR_Y_pred_sorted]
Resplot = Resplot(10:end-10,:)

% Plotting the new residuals
figure('Name','Model residual, some outliers removed')
scatter(Resplot(:,1),Resplot(:,2))
xlabel('Residual')
ylabel('Predicted')


% Calculating R-squared in test set
Model_LR_Rsquared = 1 - (Model_LR_RMSE_test/std(Y_test - mean(Model_LR_Y_pred)));

% Plotting the result
figure('Name','Linear Regression Result')
for i = 1:data_columns-1
    subplot(2,2,i)
    scatter(X_test(:,i),Y_test,0.8)
    hold on
    scatter(X_test(:,i),Model_LR_Y_pred, 0.8, 'r')
    legend('Actual','Predicted')
    xlabel(data_raw.Properties.VariableNames(i))
    ylabel('PE')
    hold off
end

%% Model evaluation - Random forest regression
% Choosing optimal model based on RMSE
[Model_RF_RMSE_min,I] = min(Model_RF_RMSE_test(:));
[Model_RF_Optimal_Trees, Model_RF_Optimal_NumPredSample] = ind2sub(size(Model_RF_RMSE_test),I);

% Discarding other RF models to save memory
Model_RF = TreeBagger(Model_RF_Optimal_Trees,X_train,Y_train,'Method','Regression','NumPredictorsToSample',Model_RF_Optimal_NumPredSample)

% Calculating the performance of the optimal model in the training set
Model_RF_Y_train = predict(Model_RF,X_train);
Model_RF_RMSE_train = sqrt(mean((Y_train - Model_RF_Y_train).^2));
Model_RF_Rsquared_train = 1 - (Model_RF_RMSE_train/std(Y_test - mean(Model_RF_Y_pred)));

% Calculating R-squared in test set
Model_RF_Rsquared_test = 1 - (Model_RF_RMSE_min/std(Y_test - mean(Model_RF_Y_pred)));

% Plotting the result
figure('Name','Random Forest Regression Result')
for j = 1:4
    subplot(2,2,j)
    scatter(X_test(:,j),Y_test,0.8)
    hold on
    scatter(X_test(:,j),Model_RF_Y_pred, 0.8, 'r')
    hold off
end

% Plotting the RMSE vs number of trees and number of variables to sample
figure('Name','RMSE by number of trees and number of variables to sample')
s = surf(Model_RF_RMSE_test,'FaceAlpha',0.5)
s.EdgeColor = 'none';
ylabel('Number of trees')
xlabel('Number of variables to sample')
zlabel('Test RMSE')

fprintf('The minimum RMSE obtained is %.4f with a model of %.0f tree(s) taking %.0f predictor(s) to the sample\n',Model_RF_RMSE_min, Model_RF_Optimal_Trees, Model_RF_Optimal_NumPredSample)

%% Further result 1 - Varying the hold out ratio to 0.3
% Initialising training set and test set
hold_out_ratio = 0.3;                                      %%%Customisation
FR1_train_size = round(data_rows*(1-hold_out_ratio));
FR1_test_size = data_rows - FR1_train_size;

FR1_X_train = data_normalised(1:FR1_train_size,1:data_columns-1);
FR1_Y_train = data_normalised(1:FR1_train_size,end);
FR1_X_test = data_normalised(1:FR1_test_size,1:data_columns-1);
FR1_Y_test = data_normalised(1:FR1_test_size,end);

% Fitting and evaluating LR model
FR1_Model_LR = fitlm(FR1_X_train, FR1_Y_train,'RobustOpts', 'off')
FR1_Model_LR_Y_pred = predict(FR1_Model_LR,FR1_X_test);
FR1_Model_LR_RMSE_test = sqrt(mean((FR1_Y_test - FR1_Model_LR_Y_pred).^2));

% Fitting RF model
rng(2) % For reproducibility
FR1_Model_RF = TreeBagger(Model_RF_Optimal_Trees,FR1_X_train,FR1_Y_train,'Method','Regression','NumPredictorsToSample',Model_RF_Optimal_NumPredSample)
FR1_Model_RF_Y_pred = predict(FR1_Model_RF,FR1_X_test);
FR1_Model_RF_RMSE_test_ = sqrt(mean((FR1_Y_test - FR1_Model_RF_Y_pred).^2));

%% Further result 2 - Varying the hold out ratio to 0.1
% Initialising training set and test set                   
hold_out_ratio = 0.1;                                      %%%Customisation
FR2_train_size = round(data_rows*(1-hold_out_ratio));
FR2_test_size = data_rows - FR2_train_size;

FR2_X_train = data_normalised(1:FR2_train_size,1:data_columns-1);
FR2_Y_train = data_normalised(1:FR2_train_size,end);
FR2_X_test = data_normalised(1:FR2_test_size,1:data_columns-1);
FR2_Y_test = data_normalised(1:FR2_test_size,end);

% Fitting and evaluating LR model
FR2_Model_LR = fitlm(FR2_X_train, FR2_Y_train,'RobustOpts', 'off')
FR2_Model_LR_Y_pred = predict(FR2_Model_LR,FR2_X_test);
FR2_Model_LR_RMSE_test = sqrt(mean((FR2_Y_test - FR2_Model_LR_Y_pred).^2));

% Fitting RF model
rng(2) % For reproducibility
FR2_Model_RF = TreeBagger(Model_RF_Optimal_Trees,FR2_X_train,FR2_Y_train,'Method','Regression','NumPredictorsToSample',Model_RF_Optimal_NumPredSample)
FR2_Model_RF_Y_pred = predict(FR2_Model_RF,FR2_X_test);
FR2_Model_RF_RMSE_test_ = sqrt(mean((FR2_Y_test - FR2_Model_RF_Y_pred).^2));

%% Further result 3 - Removing AP from the model
% Removing the selected feature
FR3_data_normalised = data_normalised(:,[1 2 4 5])         %%%Customisation

% Initialising training set and test set
hold_out_ratio = 0.2;                                      %%%Customisation
[data_rows,data_columns] = size(FR3_data_normalised);
FR3_train_size = round(data_rows*(1-hold_out_ratio));
FR3_test_size = data_rows - FR3_train_size;

FR3_X_train = FR3_data_normalised(1:FR3_train_size,1:data_columns-1);
FR3_Y_train = FR3_data_normalised(1:FR3_train_size,end);
FR3_X_test = FR3_data_normalised(1:FR3_test_size,1:data_columns-1);
FR3_Y_test = FR3_data_normalised(1:FR3_test_size,end);

% Fitting and evaluating LR model
FR3_Model_LR = fitlm(FR3_X_train, FR3_Y_train,'RobustOpts', 'off')
FR3_Model_LR_Y_pred = predict(FR3_Model_LR,FR3_X_test);
FR3_Model_LR_RMSE_test = sqrt(mean((FR3_Y_test - FR3_Model_LR_Y_pred).^2));

% Fitting RF model
rng(2) % For reproducibility
FR3_Model_RF = TreeBagger(Model_RF_Optimal_Trees,FR3_X_train,FR3_Y_train,'Method','Regression','NumPredictorsToSample',Model_RF_Optimal_NumPredSample)
FR3_Model_RF_Y_pred = predict(FR3_Model_RF,FR3_X_test);
FR3_Model_RF_RMSE_test_ = sqrt(mean((FR3_Y_test - FR3_Model_RF_Y_pred).^2));

%% Further result 4 - Removing RH from the model
% Removing the selected feature
FR4_data_normalised = data_normalised(:,[1 2 3 5])         %%%Customisation

% Initialising training set and test set
hold_out_ratio = 0.2;                                      %%%Customisation
[data_rows,data_columns] = size(FR4_data_normalised);
FR4_train_size = round(data_rows*(1-hold_out_ratio));
FR4_test_size = data_rows - FR4_train_size;

FR4_X_train = FR4_data_normalised(1:FR4_train_size,1:data_columns-1);
FR4_Y_train = FR4_data_normalised(1:FR4_train_size,end);
FR4_X_test = FR4_data_normalised(1:FR4_test_size,1:data_columns-1);
FR4_Y_test = FR4_data_normalised(1:FR4_test_size,end);

% Fitting and evaluating LR model
FR4_Model_LR = fitlm(FR4_X_train, FR4_Y_train,'RobustOpts', 'off')
FR4_Model_LR_Y_pred = predict(FR4_Model_LR,FR4_X_test);
FR4_Model_LR_RMSE_test = sqrt(mean((FR4_Y_test - FR4_Model_LR_Y_pred).^2));

% Fitting RF model
rng(2) % For reproducibility
FR4_Model_RF = TreeBagger(Model_RF_Optimal_Trees,FR4_X_train,FR4_Y_train,'Method','Regression','NumPredictorsToSample',Model_RF_Optimal_NumPredSample)
FR4_Model_RF_Y_pred = predict(FR4_Model_RF,FR4_X_test);
FR4_Model_RF_RMSE_test_ = sqrt(mean((FR4_Y_test - FR4_Model_RF_Y_pred).^2));

%% Further result 5 - Removing AP and RH from the model
% Removing the selected feature
FR5_data_normalised = data_normalised(:,[1 2 5])           %%%Customisation

% Initialising training set and test set
hold_out_ratio = 0.2;                                      %%%Customisation
[data_rows,data_columns] = size(FR5_data_normalised);
FR5_train_size = round(data_rows*(1-hold_out_ratio));
FR5_test_size = data_rows - FR5_train_size;

FR5_X_train = FR5_data_normalised(1:FR5_train_size,1:data_columns-1);
FR5_Y_train = FR5_data_normalised(1:FR5_train_size,end);
FR5_X_test = FR5_data_normalised(1:FR5_test_size,1:data_columns-1);
FR5_Y_test = FR5_data_normalised(1:FR5_test_size,end);

% Fitting and evaluating LR model
FR5_Model_LR = fitlm(FR5_X_train, FR5_Y_train,'RobustOpts', 'off')
FR5_Model_LR_Y_pred = predict(FR5_Model_LR,FR5_X_test);
FR5_Model_LR_RMSE_test = sqrt(mean((FR5_Y_test - FR5_Model_LR_Y_pred).^2));

% Fitting RF model
rng(2) % For reproducibility
FR5_Model_RF = TreeBagger(Model_RF_Optimal_Trees,FR5_X_train,FR5_Y_train,'Method','Regression','NumPredictorsToSample',Model_RF_Optimal_NumPredSample)
FR5_Model_RF_Y_pred = predict(FR5_Model_RF,FR5_X_test);
FR5_Model_RF_RMSE_test_ = sqrt(mean((FR5_Y_test - FR5_Model_RF_Y_pred).^2));