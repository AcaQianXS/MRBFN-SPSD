function [cost, gradient] = MRBFN_SoftmaxLoss_DynamicFeaturesElimination(theta,featureEliminationArray, DO, hidden_neuron_num,input_train)
%output weights+feature weights+spreads+centers

[input_dim,Pattern_num] = size(input_train);
output_dim = size(DO,1);

weight_theta = reshape( theta(1:(hidden_neuron_num+1)*output_dim),output_dim,hidden_neuron_num+1);
featureWeights_theta = theta((hidden_neuron_num+1)*output_dim+1:(hidden_neuron_num+1)*output_dim+input_dim);
spread_theta = theta((hidden_neuron_num+1)*output_dim+input_dim+1:(hidden_neuron_num+1)*output_dim+input_dim+hidden_neuron_num);
center_theta = reshape( theta( (hidden_neuron_num+1)*output_dim+input_dim+hidden_neuron_num+1:end),input_dim,hidden_neuron_num);


reduced_input_train = input_train;
reduced_input_train(featureEliminationArray,:)=[];
reduced_center_theta = center_theta;
reduced_center_theta(featureEliminationArray,:)=[];
reduced_featureWeights_theta = featureWeights_theta;
reduced_featureWeights_theta(featureEliminationArray) = [];
reduced_input_dim = length(reduced_featureWeights_theta);

OH_tr = zeros(hidden_neuron_num,Pattern_num);
train_r2 = zeros(hidden_neuron_num,Pattern_num);
for k=1:hidden_neuron_num
    train_r2(k,:) = sum((reduced_featureWeights_theta.*bsxfun(@minus,reduced_input_train,reduced_center_theta(:,k))).^2,1);
    OH_tr(k,:) = exp( -train_r2(k,:) ./ (2* (spread_theta(k)).^2 ) );
end

hidden_response_withBias = [OH_tr;ones(1,Pattern_num)];
hidden_response_T = OH_tr.';

network_output = weight_theta * hidden_response_withBias;%output_dim * Pattern_num 
network_output_for_p = bsxfun(@minus, network_output, max(network_output));
probability = bsxfun(@rdivide, exp(network_output_for_p ), sum(exp(network_output_for_p )));%probability matrix
cost = -1/Pattern_num * DO(:)' * log(probability(:));

weight_gradient = -1/Pattern_num * (DO - probability) *  hidden_response_withBias.';%output_dim*hidden_neuron_num+1
weight_gradient = reshape(weight_gradient,output_dim*(hidden_neuron_num+1),1);

distance_square_divide_spread3 = bsxfun(@rdivide, train_r2, spread_theta.^3);
spread_gradient = zeros(1,hidden_neuron_num);

DO_T = DO.';
probability_T = probability.';

for m=1:output_dim
    spread_gradient = spread_gradient +  sum(weight_theta (m,1:hidden_neuron_num).*(( DO_T(:,m) - probability_T(:,m) ) ).*((distance_square_divide_spread3.').*hidden_response_T));   
end
spread_gradient = -spread_gradient / Pattern_num ;

spread_theta_T = spread_theta.';

reduced_center_gradient = zeros(reduced_input_dim,hidden_neuron_num);
temp_weight_theta = weight_theta(:,1:hidden_neuron_num);
for p=1:Pattern_num     
     reduced_center_gradient = reduced_center_gradient + (reduced_featureWeights_theta.^2).*bsxfun(@times,reduced_input_train(:,p)-reduced_center_theta,( DO_T(p,:) - probability_T(p,:) )* temp_weight_theta.* (1./(spread_theta_T).^2) .* hidden_response_T(p,:));
end
center_gradient = zeros(input_dim,hidden_neuron_num);
center_gradient(~(featureEliminationArray),:) = reduced_center_gradient;
center_gradient = -center_gradient(:)/Pattern_num;

featureWeights_gradient = zeros(input_dim,1);

for i=1:input_dim
    if(featureEliminationArray(i))
        featureWeights_gradient(i) = 0;
    else
        temp_center_theta = center_theta(i,:);
        for p=1:Pattern_num 
          featureWeights_gradient(i) = featureWeights_gradient(i) + featureWeights_theta(i)*( DO_T(p,:) - probability_T(p,:) )*(temp_weight_theta * ( OH_tr(:,p).* ( (input_train(i,p)-temp_center_theta).').^2 ./ ((spread_theta).^2) ));      
        end
    end
end

featureWeights_gradient = featureWeights_gradient/Pattern_num;
gradient = [weight_gradient;featureWeights_gradient;spread_gradient.';center_gradient];

end