function [test_right,nonZeroWeightsNum,center_indices,predicted_classes,network_outputs] = MRBFN_SPSD_test(w,funTest,test_labels,outputWeights_index_end,output_dim)

network_outputs =  funTest(w);

[~,predicted_classes]=max(network_outputs,[],1);
predicted_classes = predicted_classes.';
test_right = sum(test_labels == predicted_classes);

num_hidden_neuron = outputWeights_index_end/output_dim;%include the bias
output_weights = reshape( w(1:outputWeights_index_end),output_dim,num_hidden_neuron);
nonZeroWeightsNum = nnz( output_weights );

center_indices = find(sum(output_weights~=0)==0);

end