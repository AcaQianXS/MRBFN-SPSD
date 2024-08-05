clear all;
clc;
addpath(genpath(pwd));

load Thyroid_train_inst;
load Thyroid_train_labels;
load Thyroid_test_inst;
load Thyroid_test_labels;

%the RBF centers and spreads are determined using the learning rule based on the maximum spread proposed by Rouhani and Javan
%Two fast and accurate heuristic RBF learning rules for data classification[J]. Neural Networks, 2016, 75: 150-161.
load RBF_centers
load RBF_spreads
center_num = length(RBF_spreads);

training_label = Thyroid_train_labels;
test_label = Thyroid_test_labels;

training_data = Thyroid_train;
test_data = Thyroid_test;

output_dim = 3;
train_num = length(training_label);
test_num = length(test_label);

maxIterations = 10;
lambdaW = 1e-5; % L1 regularization parameter for output weights
lambdaF = 1e-5; % L1 regularization parameter for feature weights

input_dim = size(test_data,1);
DO = full(sparse(training_label, 1:train_num, 1));
pseudo_inverse_weights = pseudo_inverse(RBF_centers,RBF_spreads,training_data,center_num,train_num,training_label);

N = (center_num+1)*output_dim + input_dim + center_num + center_num*input_dim ;%output weights+feature weights+spreads+centers
theta = -0.05 + 0.1 * rand(N,1);
outputWeights_index_end = (center_num+1)*output_dim;
featureWeights_index_end = (center_num+1)*output_dim+input_dim;
theta(1:outputWeights_index_end) = pseudo_inverse_weights(:);
theta(outputWeights_index_end+1:featureWeights_index_end) = 1;
theta(featureWeights_index_end+1:featureWeights_index_end+center_num) = RBF_spreads.';
theta(featureWeights_index_end+center_num+1:end) = RBF_centers(:);

[test_right_num,num_of_pruned_centers,nonZeroWeightsNum,removed_features_indices,running_time,cost] = MRBFN_SPSD_DynamicFeaturesElimination(theta,DO,center_num,training_data,test_data,outputWeights_index_end,featureWeights_index_end,test_label,output_dim,maxIterations,lambdaW,lambdaF);
