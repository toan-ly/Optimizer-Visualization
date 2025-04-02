function [step_vec, step_size] = RMSProp(x, y, grad_x, grad_y, iter, lr, grad_thres)
% Function to perform optimization using RMSProp
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

decay_rate = 0.9;
epsilon = 1e-5;

% Persistent cache for decaying average of squared gradients
persistent G
if isempty(G)
    G = zeros(size(grad));
end

% Update history with decaying average of squared gradients
G = decay_rate * G + (1 - decay_rate) * grad.^2;

% Compute step size using RMSProp update rule
step_vec = - lr .* grad ./ (sqrt(G) + epsilon);
step_size = norm(step_vec);
step_vec = step_vec / step_size;
end
