function q_dot = boat_ode(t_current, q_current, lambda, time_vector, thruster_history, robot_params)
% =========================================================================
% boat_ode.m
% =========================================================================
% This function defines the complete state-space model q_dot = f(q, u).
% It is the mathematical core of the simulation. It calculates the time
% derivative of the state vector (q_dot) at a given time (t_current) for
% a given state (q_current).
% =========================================================================

    % --- 1. Unpack Inputs ---
    
    % Unpack the current state vector q = [x, y, psi, u, v, r]'
    % (MATLAB's ODE solvers work with column vectors)
    q_current = q_current(:); % Ensure it's a column vector
    psi = q_current(3);
    u   = q_current(4);
    v   = q_current(5);
    r   = q_current(6);
    vel_body = q_current(4:6); % [u; v; r]
    
    % Unpack the parameters to be identified
    m11 = lambda(1); m22 = lambda(2); m33 = lambda(3);
    Xu  = lambda(4); Yv  = lambda(5); Nr  = lambda(6);
    
    % Unpack robot physical constants
    a = robot_params.a;
    b = robot_params.b;
    
    % --- 2. Get Current Thruster Input ---
    % The ODE solver evaluates at arbitrary time steps (t_current). We need
    % to find the thruster commands at THIS specific time. We use linear
    % interpolation on the experimental thruster data.
    f = zeros(4,1);
    for i = 1:4
        % --- FIX: Added 'extrap' to prevent NaN errors from the ODE solver ---
        % This makes the simulation more robust if ode45 requests a time
        % slightly outside the bounds of the experimental time vector.
        f(i) = interp1(time_vector, thruster_history(:,i), t_current, 'linear', 'extrap');
    end
    
    % --- 3. Construct the System Matrices ---
    
    % Mass and Inertia Matrix (with added mass)
    M = diag([m11, m22, m33]);
    
    % Coriolis and Centripetal Matrix
    C = [0,   0,   -m22*v;
         0,   0,    m11*u;
         m22*v, -m11*u, 0];
         
    % Damping (Drag) Matrix
    D = diag([Xu, Yv, Nr]);
    
    % --- CORRECTED Thruster Configuration Matrix (v3) ---
    % This version now EXACTLY matches the definition provided in the
    % source documentation (Equation 7).
    B = [1,    1,    0,    0;      % Surge force (tau_1) from f1, f2
         0,    0,    1,    1;      % Sway force (tau_2) from f3, f4
         a/2, -a/2,  b/2, -b/2];  % Torque (tau_3)
         
    % Rotation Matrix (Body to Inertial)
    R_psi = [cos(psi), -sin(psi), 0;
             sin(psi),  cos(psi), 0;
             0,         0,        1];
    
    % --- 4. Calculate the State Derivatives ---
    
    % Calculate total forces and torques from thrusters
    tau = B * f;
    
    % Calculate v_dot (body-frame accelerations) using dynamics eq. (9)
    % M*v_dot = tau - (C+D)*v  => v_dot = M\(tau - (C+D)*v)
    v_dot = M \ (tau - (C + D) * vel_body);
    
    % Calculate eta_dot (inertial-frame velocities) using kinematics eq. (8)
    eta_dot = R_psi * vel_body;
    
    % Combine to form the full state derivative vector
    q_dot = [eta_dot; v_dot];
    
end
