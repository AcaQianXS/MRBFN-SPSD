function [test_right_num,predicted_classes,num_of_pruned_centers,nonZeroWeightsNum,removed_features_indices,f,featureWeights,network_outputs] = MRBFN_SPSD_DynamicFeaturesElimination(w,labelMatrix,center_num,training_data,test_data,outputWeights_index_end,featureWeights_index_end,test_labels,output_dim,maxIter,lambdaOutPutW,lambdaFeatureW)

featureZeroOutFlagArray = zeros(featureWeights_index_end-outputWeights_index_end,1);
funObj = @(w,featureZeroOutFlagArray)MRBFN_SoftmaxLoss_DynamicFeaturesElimination(w,featureZeroOutFlagArray,labelMatrix,center_num,training_data);
funTest = @(w)MRBFN_SoftmaxOutput_FeatureSelection(w,output_dim,center_num,test_data);

% [verbose,optTol,progTol,maxIter,suffDec,corrections,K] = ...
%     myProcessOptions(options,'verbose',1,'optTol',1e-9,'progTol',1e-9,...
%     'maxIter',myMaxIterations,'suffDec',1e-4,'corrections',10,'K',[]); 

verbose = 1;
optTol = 1e-9;
progTol = 1e-9;
suffDec = 1e-4;
corrections = 10; % for L-BFGS
K_all = featureWeights_index_end; %the total number of output weights and feature weights

if verbose
    fprintf('%6s %6s %12s %12s %12s %6s %6s\n','Iter','fEvals','stepLen','fVal','optCond','nnz','addStp');
end

parametersNum = length(w);
L1_lambda = [ones(outputWeights_index_end,1)*lambdaOutPutW;ones(featureWeights_index_end-outputWeights_index_end,1)*lambdaFeatureW;zeros(parametersNum-featureWeights_index_end,1)];

% Evaluate Initial Point
p = length(w);

[f,g] = pseudoGrad(funObj,w,featureZeroOutFlagArray,L1_lambda);
funEvals = 1;    


% Check optimality
optCond = max(abs(g));
if optCond < optTol
    if verbose
        fprintf('First-order optimality satisfied at initial point\n');
    end
    return;
end

currentIter=1;
% Main loop
while currentIter <=maxIter
    
    % Compute preliminary working set (non-zero and unregularized parameters)
    W = L1_lambda == 0 | w ~=0;
    
    % Compute direction
    d = zeros(p,1);

    if currentIter == 1
        [sorted,sortedInd] = sort(abs(g(1:K_all)),'descend');
        W(sortedInd) = 1;   
        d(W) = -g(W);
        t = 1;
        Y = zeros(p,0);
        S = zeros(p,0);
        sigma = 1;
        
    else
        y = g-g_old;
        s = w-w_old;
        
        correctionsStored = size(Y,2);
        if correctionsStored < corrections
            Y(:,correctionsStored+1) = y;
            S(:,correctionsStored+1) = s;
        else
            Y = [Y(:,2:corrections) y];
            S = [S(:,2:corrections) s];
        end
        
        ys = y'*s;
        if ys > 1e-10
            sigma = ys/(y'*y);
        end
       
        curvSat = sum(Y(W,:).*S(W,:)) > 1e-10; 
        
        d(W) = lbfgsC(-g(W),S(W,curvSat),Y(W,curvSat),sigma);                                             
        t = 1;
        
        % The current d is fine, now do a binary search for a value  
        % k <= K such that the sign condition is satisfied for k but not for k+1
        [sorted,sortedInd] = sort( abs(g(1:K_all) ),'descend');
        LB = 0;
        K = sum(double(L1_lambda ~= 0 & w == 0 & g~=0));
        UB = K+1;
        while UB-LB ~= 1
           k = ceil((UB+LB)/2);
           if g(sortedInd(k)) == 0 %sortedInt is sorted according to the absolute value of pesudo-gradient, if g(sortedInd(k)) == 0, sortedInd(k+1:UB)==0
              %fprintf('Variable should not move away from zero\n');
              UB = k;
           else
               W_new = W;
               W_new(sortedInd(1:k)) = 1;
               d_new = zeros(p,1);
               curvSat = sum(Y(W_new,:).*S(W_new,:)) > 1e-10;
               d_new(W_new) = lbfgsC(-g(W_new),S(W_new,curvSat),Y(W_new,curvSat),sigma);
               
               if any(sign(d_new(w==0)).*sign(-g(w==0)) == -1) 
                   %fprintf('Sign condition violated\n');
                   UB = k;
               else
                   %fprintf('Sign condition satisfied\n');
                   d = d_new;
                   LB = k;
               end
               
           end
        end
    end
    f_old = f;
    g_old = g;
    w_old = w;
    
    % Compute desired orthant
    xi = sign(w);
    xi(w==0) = sign(-g(w==0));%for zero-valued variables, desired orthant contains the sign of the negative gradient 
                              %for non-zero variables, desired orthant contains the same sign of theirs
    
    gtd = g'*d;
    
    % Compute projected point
    w_new(1:featureWeights_index_end,1) = orthantProject(w(1:featureWeights_index_end)+t*d(1:featureWeights_index_end),xi(1:featureWeights_index_end));%xi represents the desired orthant Po projection;
    w_new(featureWeights_index_end+1:parametersNum,1) = w(featureWeights_index_end+1:parametersNum)+t*d(featureWeights_index_end+1:parametersNum);

    featureZeroOutFlagArray = zeros(featureWeights_index_end-outputWeights_index_end,1);
    featureZeroOutFlagArray(w_new(outputWeights_index_end+1:featureWeights_index_end,1) == 0) = 1;
     
    [f_new,g_new] = pseudoGrad(funObj,w_new,featureZeroOutFlagArray,L1_lambda);
    funEvals = funEvals+1;
   
    % Line search along projection arc
    while f_new > f + suffDec*g'*(w_new-w) || ~isLegal(f_new) %Armijo condition
        t_old = t;
        
        % Backtracking
        if ~isLegal(f_new)
            t = .5*t;
        else
            t = polyinterp([0 f gtd; t f_new g_new'*d]);
        end
        
        % Adjust if interpolated value near boundary
        if t < t_old*1e-3
            t = t_old*1e-3;
        elseif t > t_old*0.6
            t = t_old*0.6;
        end
        
        % Check whether step has become too small
        if max(abs(t*d)) < progTol
            t = 0;
            w_new = w;
            f_new = f;
            g_new = g;
            break;
        end
        
        % Compute projected point
        w_new(1:featureWeights_index_end,1) = orthantProject(w(1:featureWeights_index_end)+t*d(1:featureWeights_index_end),xi(1:featureWeights_index_end));%xi represents the desired orthant Po projection;
        w_new(featureWeights_index_end+1:parametersNum,1) = w(featureWeights_index_end+1:parametersNum)+t*d(featureWeights_index_end+1:parametersNum);

        featureZeroOutFlagArray = zeros(featureWeights_index_end-outputWeights_index_end,1);
        featureZeroOutFlagArray(w_new(outputWeights_index_end+1:featureWeights_index_end,1) == 0) = 1;
        [f_new,g_new] = pseudoGrad(funObj,w_new,featureZeroOutFlagArray,L1_lambda);
        funEvals = funEvals+1;
    end
    
    % Take step
    w = w_new;
    f = f_new;
    g = g_new;
    
    currentIter = currentIter+1;
    if verbose
        fprintf('%6d %6d %8.5e %8.5e %8.5e %6d\n',currentIter,funEvals,t,f,max(abs(g)),nnz(w));
    end
    
    % Check Optimality
    optCond = max(abs(g));
    if optCond < optTol
        if verbose
            fprintf('First-order optimality below optTol\n');
        end
        break;
    end
    
    % Check for lack of progress
    if max(abs(t*d)) < progTol || abs(f-f_old) < progTol
        if verbose
            fprintf('Progress in parameters or objective below progTol\n');
        end
        break;
    end
    
end

featureWeights = w(outputWeights_index_end+1:featureWeights_index_end);
removed_features_indices = find((featureWeights == 0) == 1).';
[test_right_num,nonZeroWeightsNum,center_index_for_prune,predicted_classes,network_outputs] = MRBFN_SPSD_test(w_new,funTest,test_labels,outputWeights_index_end,output_dim);
num_of_pruned_centers  = length(center_index_for_prune);    

fprintf('\n nonZeroWeightsNum=%d\t num_of_pruned_centers=%d\t num_of_removed_features=%d\t L1_test_accuracy=%f\t cost=%f\t at iteration %d\n',nonZeroWeightsNum,length(center_index_for_prune),length(removed_features_indices), test_right_num/length(test_labels),f,currentIter);

end

% Psuedo-gradient calculation
function [f,pGrad] = pseudoGrad(funObj,w,featureZeroOutFlagArray,lambda)

[f,g] = funObj(w,logical(featureZeroOutFlagArray));

f = f + sum(lambda.*abs(w));

pGrad = zeros(size(g));
pGrad(g < -lambda) = g(g < -lambda) + lambda(g < -lambda);
pGrad(g > lambda) = g(g > lambda) - lambda(g > lambda);
nonZero = w~=0 | lambda==0;
pGrad(nonZero) = g(nonZero) + lambda(nonZero).*sign(w(nonZero));

end

function [w] = orthantProject(w,xi)
w(sign(w) ~= xi) = 0;
end
