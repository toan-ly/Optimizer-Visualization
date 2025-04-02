function [step_vec, step_size] = AdaGrad(x, y, grad_x, grad_y, iter, lr, grad_thres)
% Function to perform optimization using AdaGrad
% Input:
%   x, y: current position
%   grad_x, grad_y: gradients of the objective function at (x, y)
%   iter: number of iterations
%   lr: learning rate
% Output:
%   step_vec: vector representing the step taken in each dimension
%   step_size: magnitude of the step

grad = [grad_x, grad_y];
if norm(grad) < grad_thres
    step_vec = [0, 0];
    step_size = 0;
    return;
end

p_old = [x, y];
epsilon = 1e-5;

persistent G
if isempty(G)
    G = zeros(size(grad));
end

% Update history with squared gradients
G = G + grad.^2;

% Compute step size using AdaGrad update rule
p_new = p_old - lr .* grad ./ (sqrt(G) + epsilon);

% AdaGrad update with epsilon for numerical stability
step_vec = p_new - p_old;
step_size = norm(step_vec);
step_vec = step_vec / step_size;
end
