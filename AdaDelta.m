function [step_vec, step_size] = AdaDelta(x, y, gradX, gradY, iter, lr, grad_threshold)
    grad = [gradX, gradY];
    if norm(grad) < grad_threshold
        step_vec = [0, 0]; 
        step_size = 0;
        return;
    end

    % AdaDelta hyperparameters
    gamma = 0.9; % Decay rate
    epsilon = 1e-5; % Small constant to avoid division by zero
    
    persistent E_grad E_delta;
    if isempty(E_grad)
        E_grad = [0, 0];
        E_delta = [0, 0];
    end
        
    % Update E_grad with exponentially decaying average of squared gradients
    E_grad = gamma * E_grad + (1 - gamma) * grad.^2;
    
    % Calculate RMS of previous deltas
    RMS_delta = sqrt(E_delta + epsilon);
    
    % Calculate RMS of gradients
    RMS_grad = sqrt(E_grad + epsilon);
    
    % Calculate update step
    update_step = -(RMS_delta ./ RMS_grad) .* grad;
    
    % Update E_delta with exponentially decaying average of squared updates
    E_delta = gamma * E_delta + (1 - gamma) * update_step.^2;
    
    % Update position
    step_vec = update_step;
    
    % Calculate step size
    step_size = lr;
    

end
