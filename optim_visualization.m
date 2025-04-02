clear;  % Clear workspace variables
close all;  % Close all figures

% Define optimizers and their learning rates
[optimizers, optimizer_names, learning_rates, colors] = defineOptimizers();
num_optimizers = numel(optimizers);

% Define parameters
grad_thres = 1e-6;
thres = 0.5;
total_iter = 200;
offset_points = 5;
offset_texts = 50;
num_reps = 1;
map_size = 100;

% Create map and gradient functions
[map_functions, dif_func_x, dif_func_y, global_minimums] = createMaps(map_size);
num_funcs = length(map_functions);

% Generate meshgrid
[X, Y] = generateMeshGrid(map_size);

% winning_optim = '';

for idx = 1:num_funcs
    % Create world
    world = map_functions{idx}(X, Y);

    for rep = 1:num_reps
        filename = sprintf('Animations/World_%d_Rep_%d.gif', idx, rep);

        % Initialize points
        p_positions = initializePoints(map_size, num_optimizers);

        % Display world
        hf = figure;
        set(hf, 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);

        % Subplot for 3D visualization
        h1 = subplot(3, 1, [1,2]);
        surf(X, Y, world, 'FaceAlpha', 0.9);
        colormap('parula(15)');
        colorbar;
        % axis off;
        grid on;
        hold on

        % Plot global minimum
        global_min = global_minimums{idx};
        plot3(global_min(1), global_min(2), map_functions{idx}(global_min(1), global_min(2) + offset_points), ...
            'p', 'Markersize', 30, 'Color', 'w', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k');

        % Plot initial player positions
        h_points = gobjects(num_optimizers, 1);
        h_texts = gobjects(num_optimizers, 1);
        for p = 1:num_optimizers
            h_points(p) = plot3(p_positions(p, 1), p_positions(p, 2), map_functions{idx}(p_positions(p, 1), p_positions(p, 2) + offset_points), ...
                'o', 'Markersize', 15, 'Color', colors{p}, 'MarkerFaceColor', colors{p}, 'MarkerEdgeColor', 'k');

            h_texts(p) = text(p_positions(p, 1), p_positions(p, 2), map_functions{idx}(p_positions(p, 1), p_positions(p, 2)) + offset_texts, ...
                optimizer_names{p}, 'VerticalAlignment', 'baseline', 'Color', colors{p}, 'FontWeight', 'bold', 'FontSize', 12);
        end
        title_str = ['World ', num2str(idx), ' Rep ', num2str(rep)];
        htitle = title( title_str, 'FontSize', 12, 'FontWeight', 'bold' );
        xlabel('X', 'FontSize', 10, 'FontWeight', 'bold');
        ylabel('Y', 'FontSize', 10, 'FontWeight', 'bold');
        zlabel('Loss', 'FontSize', 10, 'FontWeight', 'bold');

        % Subplot for loss visualization
        h2 = subplot(3, 1, 3);
        h_plots = gobjects(num_optimizers, 1);
        hold on
        for p = 1:num_optimizers
            h_plots(p) = plot(nan(1, total_iter), '.-', 'LineWidth', 1, 'Color', colors{p});
        end

        xlabel('Iteration', 'FontSize', 10, 'FontWeight', 'bold');
        ylabel('Loss', 'FontSize', 10, 'FontWeight', 'bold');
        hold on
        plot([0, total_iter], [thres, thres], 'k--', 'LineWidth', 1);
        view(h1, [-102, 45]);
        % view(h1, [-90, 90]);

        legend_names = cell(1, num_optimizers);
        for i = 1:num_optimizers
            legend_names{i} = sprintf('%s, %g', optimizer_names{i}, learning_rates(i));
        end
        legend(h2, legend_names, 'FontSize', 12, 'Location', 'northeastoutside');
        % legend(h2, [optimizer_names, ', ', learning_rates], 'FontSize', 10, 'Location', 'northeast');

        losses = zeros(num_optimizers, total_iter);

        % Loop through iterations
        for iter = 1:total_iter
            % Update player positions and calculate losses
            for p = 1:num_optimizers
                p_gradX = dif_func_x{idx}(p_positions(p, 1), p_positions(p, 2));
                p_gradY = dif_func_y{idx}(p_positions(p, 1), p_positions(p, 2));
                [p_vec, p_step] = optimizers{p}(p_positions(p, 1), p_positions(p, 2), p_gradX, p_gradY, iter, learning_rates(p), grad_thres);

                % Update position
                new_positions = p_positions(p, :) + p_vec * p_step;

                % Check if out of bounds
                p_positions(p, :) = checkBounds(new_positions, map_size);

                losses(p, iter) = map_functions{idx}(p_positions(p, 1), p_positions(p, 2));
            end

            % Update plots
            for p = 1:num_optimizers
                set(h_points(p), 'XData', p_positions(p, 1), 'YData', p_positions(p, 2), ...
                    'ZData', map_functions{idx}(p_positions(p, 1), p_positions(p, 2)) + offset_points);
                set(h_texts(p), 'Position', [p_positions(p, :), map_functions{idx}(p_positions(p, 1), p_positions(p, 2)) + offset_texts]);
                set(h_plots(p), 'YData', losses(p, 1:iter));
            end
           
            % % Check for convergence
            % if any(losses(:, iter) < thres)
            %     winning_index = find(losses(:, iter) < thres, 1);
            %     winning_optim = optimizer_names{winning_index};
            % end
            
            if all(losses(:, iter) < thres)
                break; % Exit the loop if all optimizers have converged
            end

            set( htitle, 'String', [title_str, ' Iter: ', num2str(iter)] );
            pause(1e-5);
            
            % exportgraphics(gcf, filename, Append=true);
        end

        % if ~isempty(winning_optim)
        %     disp([winning_optim, ' wins!']);
        % else
        %     disp('No optimizer converged below the threshold!');
        % end

        % Ask for continuation
        % str = input('Continue? (y/n): ', 's');
        % if strcmp(str, 'n')
        %     break;
        % end
            
        close(hf);
    end
end

%-----------------------------------------------------------------------------------------------------

function [optimizers, optimizer_names, learning_rates, colors] = defineOptimizers()
    % Clearing previously defined optimizers
    clear Momentum NAG AdaGrad AdaDelta RMSProp Adam FTRL Nadam AdamW Ranger;

    addpath('Optimizers');

    % Define optimizers and learning rates
    optimizers = {@GradDescent, @Momentum, @NAG, @AdaGrad, @AdaDelta, @RMSProp, @Adam, @Nadam, @AdamW, @Ranger};
    optimizer_names = {'GradDescent', 'Momentum', 'NAG', 'AdaGrad', 'AdaDelta', 'RMSProp', 'Adam', 'Nadam', 'AdamW', 'Ranger'};
    % learning_rates = [5, 3, 3, 10, 200, 1.5, 5, 5, 5, 5];
    % learning_rates = [0.05, 0.01, 0.01, 0.1, 50, 0.02, 0.05, 0.05, 0.05, 0.05];
    learning_rates = [0.5, 0.1, 0.1, 5, 100, 1, 3, 3, 3, 1];

    % Marker colors for plotting
    colors = {'#b82f27', '#D95319', '#EDB120', '#6eff40', '#A2142F', '#7E2F8E', '#4DBEEE', '#FFFF00', '#FF00FF', '#53968d'};
    
end

% Function to create map and gradient functions
function [map_functions, dif_func_x, dif_func_y, global_minimums] = createMaps(map_size)
    % Define map functions
    map_functions = {         
        @(x, y) ((x - 30).^2 + (y - 68).^2) / 1e1, ...
        @(x, y) (sind(9 * x)/2 + cosd(13 * y)/1.5 + (x - 20).^2 / 1e3 + (y-15).^2 /1e3 + 0.6951) * 6e1, ...
        @(x, y) (sind(7 * x)/1.2 + cosd(8 * y)/2 + (x - 30).^2 / 1e3 + (y-30).^2 /1e3 + 1.1766) * 1e1, ...
        @(x, y) (sind(6 * y) / 1.5 + cosd(15 * x) / 2 + (x - 30).^2 / 1e3 + (y-50).^2 /1e3 + 1.111) * 1e2, ...
        @(x, y) (sind(15 * y) / 2 + cosd(10 * x) / 1.2 + (x - 40).^2 / 1e3 + (y-30).^2 /1e3 + 1.0136) * 1e2, ...
        @(x, y) (sind(9 * y) * 0.9 + cosd(10 * x) * 0.9 + (x - 20).^2 / 1e3 + (y-20).^2 /1e3 + 1.4375) * 1e2, ...
        @(x, y) (sind(12 * y) * 1 + cosd(10 * x) * 0.5 + (x - 40).^2 * 2 / 1e3 + (y-40).^2 * 2 /1e3 + 0.9001) * 6e1, ...
        @(x, y) (sind(((x - 5).^2 + (y - 40).^2) / 1e1) + ((x - 20).^2 + (y - 45).^2) / 1e3 - 0.1596) * 1e2, ...
        };

    % Create gradient functions
    syms x y;
    dif_func_x = cell(1, length(map_functions));
    dif_func_y = cell(1, length(map_functions));
    global_minimums = cell(1, length(map_functions));

    for i = 1:length(map_functions)
        f = map_functions{i}(x, y);
        dif_func_x{i} = matlabFunction(diff(f, x), 'vars', {x, y});
        dif_func_y{i} = matlabFunction(diff(f, y), 'vars', {x, y});

        % Find global minimum using fmincon
        options = optimoptions('particleswarm', 'Display', 'off', 'SwarmSize', 100, 'MaxIterations', 200);
        for i = 1:length(map_functions)
            [min_coords, ~] = particleswarm(@(x) map_functions{i}(x(1), x(2)), 2, [0 0], [map_size map_size], options);
            global_minimums{i} = min_coords;
        end
    end
end

% Function to initialize point positions
function p_positions = initializePoints(map_size, num_optimizers)
    initial_position = randi(map_size, 1, 2);
    initial_position = [95, 95];
    p_positions = repmat(initial_position, num_optimizers, 1);
end

% Function to generate meshgrid
function [X, Y] = generateMeshGrid(map_size)
    map_vector = 0:map_size;
    [X, Y] = meshgrid(map_vector, map_vector);
end

% Function to check bounds
function new_positions = checkBounds(positions, map_size)
    new_positions = positions;
    for dim = 1:2
        if new_positions(dim) < 0 || new_positions(dim) > map_size
            % Add a small random number to the previous position
            new_positions(dim) = positions(dim) + (randn() - 0.5) / 2 * 0.1;
        end
    end
end