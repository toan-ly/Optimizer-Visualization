function [step_vec, step_size] = GradDescent(x, y, grad_x, grad_y, iter, lr, grad_thres)
% Function to perform optimization using Gradient Descent
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

step_vec = -lr * grad;
step_size = norm(step_vec);
step_vec = step_vec / step_size; % Normalize step vec

end