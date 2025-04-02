function [step_vec, step_size] = Adam(x, y, grad_x, grad_y, iter, lr, grad_thres)
% Function to perform optimization using Adam
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
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-5;

persistent m v
if isempty(m)
    m = [0, 0];
    v = [0, 0];
end

m = beta1 * m + (1 - beta1) * grad; % Momentum
v = beta2 * v + (1 - beta2) * grad.^2; % RMS Prop

m_hat = m / (1 - beta1^iter);
v_hat = v / (1 - beta2^iter);

% Update parameters using Adam update rule
p_new = p_old - lr * m_hat ./ (sqrt(v_hat) + epsilon);

step_vec = p_new - p_old;
step_size = norm(step_vec);
step_vec = step_vec / step_size;
end
