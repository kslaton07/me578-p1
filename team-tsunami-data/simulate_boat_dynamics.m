function sim_states = simulate_boat_dynamics(lambda, time, initial_state, exp_thrusters, robot_params)
% =========================================================================
% simulate_boat_dynamics.m
% =========================================================================
% This function simulates the ASV's motion over time using a numerical
% ODE solver (ode45). It takes the model parameters (lambda), time vector,
% initial state, and the history of thruster commands as input.
% =========================================================================

    % We need to pass multiple, time-varying arguments to the ODE function
    % (lambda, thruster commands, etc.). A common and clean way to do this
    % is to wrap our ODE function in an anonymous function handle.
    ode_function_handle = @(t, q) boat_ode(t, q, lambda, time, exp_thrusters, robot_params);

    % Use ode45 to solve the differential equations
    % It takes the function handle, the time span to solve over, and the
    % initial state conditions.
    [~, sim_states] = ode45(ode_function_handle, time, initial_state);
    
end
