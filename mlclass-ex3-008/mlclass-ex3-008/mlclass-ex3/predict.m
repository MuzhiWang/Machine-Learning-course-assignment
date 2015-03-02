function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%fprintf('size(Theta1) is %d \n', size(Theta1));
%fprintf('size(Theta2) is %d \n', size(Theta2));
X = [ones(size(X, 1), 1), X];
%fprintf('size(X) is %d \n', size(X));
pTemp = X * Theta1';
%pTemp = [ones(size(pTemp, 1), 1), pTemp];
%fprintf('size(pTemp) is %d \n', size(pTemp));

z2 = sigmoid(pTemp);
z2 = [ones(size(z2, 1), 1), z2];
pTemp2 = z2 * Theta2';
%fprintf('size(pTemp2) is %d \n', size(pTemp2));
[M, I] = max(pTemp2, [], 2);

p = I;


% =========================================================================


end
