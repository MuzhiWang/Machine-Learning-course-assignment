function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


%%% J value
n = size(theta);
sum1 = 0;
sum2 = 0;

for i = 1 : m;
    z = theta' * X(i, :)';
    h = sigmoid(z);
    sum1 = sum1 + (-y(i) * log(h) - (1 - y(i)) * log (1 - h));
end
sum1 = sum1 / m;

for j = 2 : n;
    sum2 = sum2 + theta(j) ^ 2;
end
sum2 = sum2 * lambda / (2 * m);

J = sum1 + sum2;
    
%%% Gradient
for j = 1 : n;
    sum = 0;
    for i = 1 : m;
        z = theta' * X(i, :)';
        h = sigmoid(z);
        sum = sum + (h - y(i)) * X(i, j);
    end
    if j == 1;
        grad(j) = sum / m;
    else
        grad(j) = sum / m + lambda * theta(j) / m;
    end
end


% =============================================================

end
