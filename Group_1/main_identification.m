% =========================================================================
% main_identification.m
% =========================================================================
% This script runs a grey-box model identification for the ASV.
% It now implements a more robust TWO-PHASE optimization strategy to find a
% globally optimal set of parameters and avoid getting stuck in local minima.
%
% PHASE 1: GLOBAL SEARCH (using Particle Swarm)
%   - A population of 'particles' explores the entire parameter space to find
%     promising regions for the solution. This is excellent for avoiding
%     local minima.
%
% PHASE 2: LOCAL REFINEMENT (using lsqnonlin)
%   - The best point found by the particle swarm is used as a starting
%     guess for a traditional gradient-based optimizer (lsqnonlin). This
%     efficiently refines the solution to a high precision.
% =========================================================================

clear; clc; close all;

%% --- 1. Configuration ---

% --- USER: Set the path to your data files ---
data_location = "C:\Users\CKEN\Documents\MARINE_ROBOTICS\project 1\Tsunami\Group_1";
odom_file = "\2025-09-16-13-28-33-odometry-navsat.csv";

% Define the known physical parameters of the robot
robot_params.a = 0.45; % Distance between transverse thrusters (m)
robot_params.b = 0.90; % Distance between longitudinal thrusters (m)

%% --- 2. Load Experimental Data ---
fprintf('Loading experimental data...\n');
try
    % Use the data processing function to get synchronized states and forces
    [time, exp_states, exp_thrusters] = GetRoboatRuntimeData(data_location, odom_file);
    fprintf('Data loaded successfully.\n');
catch ME
    fprintf('ERROR: Could not load or process data.\n');
    rethrow(ME);
end

% The state vector q is [x, y, psi, u, v, r]'
initial_state = exp_states(1, :)'; % Get the first state as the initial condition

%% --- 3. Setup the Optimization Problem ---

% Define the parameter vector lambda = {m11, m22, m33, Xu, Yv, Nr}
% Set Lower Bounds (lb) and Upper Bounds (ub) for the parameters.
lb = [15, 15, 2,  5,  5, 1];
ub = [40, 50, 10, 30, 40, 10];
n_params = length(lb);

% The cost function for lsqnonlin (returns a vector of errors)
cost_function_handle = @(lambda) simulation_cost_function(lambda, time, exp_states, exp_thrusters, robot_params);

% The objective function for particleswarm (must return a single scalar value)
% We simply sum the squares of the errors from the cost function.
ps_objective_function = @(lambda) sum(cost_function_handle(lambda).^2);


%% --- 4. Run the Two-Phase Optimization ---

% --- PHASE 1: Global Search with Particle Swarm ---
fprintf('\n--- PHASE 1: Starting Global Search (Particle Swarm) ---\n');
fprintf('This may take a few minutes...\n');
ps_options = optimoptions('particleswarm', 'Display', 'iter', 'SwarmSize', 100, 'MaxIterations', 50);
[lambda_global_best, ~] = particleswarm(ps_objective_function, n_params, lb, ub, ps_options);

fprintf('\n--- Global Search Complete. Best point found: ---\n');
disp(lambda_global_best);

% --- PHASE 2: Local Refinement with lsqnonlin ---
fprintf('\n--- PHASE 2: Starting Local Refinement (lsqnonlin) ---\n');
fprintf('Using the result from Particle Swarm as the starting guess.\n');
lsq_options = optimoptions('lsqnonlin', 'Display', 'iter', 'StepTolerance', 1e-8, 'FunctionTolerance', 1e-8);
[lambda_identified, resnorm] = lsqnonlin(cost_function_handle, lambda_global_best, lb, ub, lsq_options);


%% --- 5. Final Results and Visualization ---

fprintf('\nOptimization finished.\n');
fprintf('\n--- Best Identified Model Parameters ---\n');
fprintf('m11 = %.2f\n', lambda_identified(1));
fprintf('m22 = %.2f\n', lambda_identified(2));
fprintf('m33 = %.2f\n', lambda_identified(3));
fprintf('Xu  = %.2f\n', lambda_identified(4));
fprintf('Yv  = %.2f\n', lambda_identified(5));
fprintf('Nr  = %.2f\n', lambda_identified(6));
fprintf('--------------------------------------\n');
fprintf('Final Residual Norm (Cost): %.4f\n', resnorm);

% Simulate the model one last time with the best parameters found
fprintf('Simulating final model for comparison...\n');
sim_states_final = simulate_boat_dynamics(lambda_identified, time, initial_state, exp_thrusters, robot_params);

% Plot the results
figure('Name', 'Model Identification Results: Experimental vs. Simulated');
labels = {'Surge Velocity (u)', 'Sway Velocity (v)', 'Yaw Rate (r)'};
for i = 1:3
    subplot(3, 1, i);
    plot(time, exp_states(:, i+3), 'b-', 'LineWidth', 2);
    hold on;
    plot(time, sim_states_final(:, i+3), 'r--', 'LineWidth', 2);
    grid on;
    title(labels{i});
    xlabel('Time (s)');
    ylabel('Velocity (m/s or rad/s)');
    legend('Experimental Data', 'Simulated Model');
end
