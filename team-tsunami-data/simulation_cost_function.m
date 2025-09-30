function error_vector = simulation_cost_function(lambda, time, exp_states, exp_thrusters, robot_params)
% =========================================================================
% simulation_cost_function.m
% =========================================================================
% This function calculates the error between the experimental velocities
% and the velocities predicted by the simulator for a given set of
% parameters (lambda). This is the function that the optimizer (lsqnonlin)
% will try to minimize.
% =========================================================================

    % The initial state for the simulation is the first state from the experiment
    initial_state = exp_states(1, :);

    % Run the simulation with the current guess for lambda
    sim_states = simulate_boat_dynamics(lambda, time, initial_state, exp_thrusters, robot_params);
    
    % Extract the experimental and simulated velocities
    % Ue = [u_e, v_e, r_e] and Us = [u_s, v_s, r_s]
    Ue = exp_states(:, 4:6); % Columns 4, 5, 6 are u, v, r
    Us = sim_states(:, 4:6);

    % Calculate the error vector ε(t) = Ue(t) - Us(t)
    error = Ue - Us;

    % --- Define the Weighting Matrix W ---
    % This diagonal matrix assigns relative importance. For now, we weigh
    % all errors equally (Identity matrix). You could increase a weight
    % to prioritize fitting a specific velocity component better.
    W = diag([1, 1, 1]); 
    
    % lsqnonlin minimizes the sum of squares of the returned vector.
    % To incorporate weights, we return W^(1/2) * error. The algorithm
    % will then minimize (W^(1/2)e)' * (W^(1/2)e) = e'We.
    % Since W is diagonal, its square root is just the sqrt of its elements.
    error_vector = error * sqrt(W);
    
    % Reshape the error matrix into a single column vector for the optimizer
    error_vector = error_vector(:);
    
end
