function network_output = MRBFN_SoftmaxOutput_FeatureSelection(theta,output_dim,hidden_neuron_num,input_train)

[input_dim,Pattern_num] = size(input_train);

weight_theta = reshape( theta(1:(hidden_neuron_num+1)*output_dim),output_dim,hidden_neuron_num+1);
featureWeights_theta = theta((hidden_neuron_num+1)*output_dim+1:(hidden_neuron_num+1)*output_dim+input_dim);
spread_theta = theta((hidden_neuron_num+1)*output_dim+input_dim+1:(hidden_neuron_num+1)*output_dim+input_dim+hidden_neuron_num);
center_theta = reshape( theta( (hidden_neuron_num+1)*output_dim+input_dim+hidden_neuron_num+1:end),input_dim,hidden_neuron_num);

O_bias = ones(1,Pattern_num);

OH_tr = zeros(hidden_neuron_num,Pattern_num);
train_r2 = zeros(hidden_neuron_num,Pattern_num);

for k=1:hidden_neuron_num
    train_r2(k,:) = sum((featureWeights_theta.*bsxfun(@minus,input_train,center_theta(:,k))).^2,1);
    OH_tr(k,:) = exp( -train_r2(k,:) ./ (2* (spread_theta(k)).^2 ) );
end

hidden_response = [OH_tr;O_bias];
network_output = weight_theta * hidden_response;%output_dim * Pattern_num

end

