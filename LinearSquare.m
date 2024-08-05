function weight = LinearSquare(H,y,lambda)
%H: hidden response
%y: target output
if nargin < 3
    lambda = 0;
end
H_square = H.'*H;
dim = length(H_square);
%tem = H_square +lambda*eye(dim);
H_square = H_square +lambda*eye(dim);

% if rank(H_square) < dim
%     disp(sprintf('ill-conditioned error!\n'));
% end
weight = H_square\(H.'*y);%A*X=B, A\B=A^-1*B
%weight = inv(tem) * (H.'*y);
end
