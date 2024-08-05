function [weights] = pseudo_inverse(centers,spreads,training_data,center_num,train_num,training_labels)

OH_tr = zeros(train_num,center_num);

for k=1:center_num
    train_r2 = sum(abs(training_data-repmat(centers(:,k),1,train_num)).^2,1);
    spreads_square = 2*spreads(k).^2;
    OH_tr(:,k) = exp( - (train_r2 / spreads_square) );
end
OH_tr = [OH_tr ones(train_num,1)];

original_DO = full(sparse(training_labels, 1:train_num, 1)).';
weights = LinearSquare(OH_tr,original_DO,0.01); 
weights = weights.';
end

