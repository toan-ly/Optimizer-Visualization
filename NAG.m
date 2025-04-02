function [step_vec, step_size] = NAG(x, y, grad_x, grad_y, iter, lr, grad_thres)
% Function to perform optimization using Nesterov Accelerated Gradient (NAG)
% Input:
%   x, y: current position
%   grad_x, grad_y: gradients of the objective function at (x, y)
%   iter: number of iterations
%   lr: learning rate
%   grad_thres: gradient threshold for convergence
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
beta = 0.9;

persistent momentum
if isempty(momentum)
    momentum = [0, 0];
end

% Update momentum
momentum = beta * momentum - lr * grad;

% Nesterov update
p_new = p_old + beta * momentum - lr * grad;

step_vec = p_new - p_old;
step_size = norm(step_vec);
step_vec = step_vec / step_size;
end
