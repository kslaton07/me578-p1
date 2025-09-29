function [time, trajectory, thrusters] = GetRoboatRuntimeData(location, odomFileName)
%[time, trajectory, thrusters] =
%GetRoboatRuntimeData(location, odomFileName)
%   
%   Helper function to process the bagged data once it has been converted
%   to a csv. All CSV files must be in the same location.
%   
%   Inputs:
%       location - string of the file's location
%       odomFileName - string of the odometry csv file
%
%   Outputs:
%       time - array of time starting from 0 (R^(Nx1))
%       trajectory - Time normalized array of the roboat states (R^(Nx6))
%       thrusters - Thruster forces (R^(Nx4))
%       
%   This file supports different frequencies for roboat odometry and 
%   forces if the trajectory is sampled at a higher frequency.
%   
%   Recommended usage of matlab `uigetfile()` to get the odometry file name
%       [odometryFile,location] = uigetfile('*-odometry-navsat.csv');


%% Get all of the necessary file names
    files = struct();
    files.odometry = odomFileName;

    dateName = split(files.odometry, '-odometry-navsat.csv');
    dateName = dateName{1};
    filesDir = dir(fullfile(location, '*.csv'));
    
    files.force = strcat(location, dateName, '-command_force.csv');    
    
    %% Get Odometry Path Data and Calculate heading angle
    odometryTable = readtable(strcat(location, files.odometry), 'Delimiter', ',');
    eul = quat2eul(table2array(odometryTable(:,[13 10 11 12])));
    odometryData = [table2array(odometryTable(:,[7 8])) eul(:,1) table2array(odometryTable(:,[15 16 20]))];

    %% Extract Thruster Forces
    forceTable = readtable(files.force, 'Delimiter', ',');
    strArr = table2array(forceTable(:,2));
    forceData = zeros([height(forceTable) 4]);
    for ii=1:height(forceTable)
        forceData(ii,:) = str2num(strArr{ii}(2:end-1));
    end

    thrusters = forceData;
    
    %% Get Time Information to sync them up 
    odomTime = datetime(odometryTable.time, 'Format', 'yyyy/MM/dd/HH:mm:ss.SSS');
    thrustTime = datetime(forceTable.time, 'Format', 'yyyy/MM/dd/HH:mm:ss.SSS');
    %% Get Trajectory Data
    trajectory = zeros([length(thrusters) 6]);
    
    % Sync the first point
    syncIdx = 1;
    while odomTime(syncIdx) < thrustTime(1)
        syncIdx = syncIdx + 1;
    end
    initInertial = odometryData(syncIdx,:);

    % Rotation matrix for resetting initial position
    rot = [cos(initInertial(3)) -sin(initInertial(3)) 0;
           sin(initInertial(3))  cos(initInertial(3)) 0;
           0 0 1] ^ -1;

    idx = 1;
    for ii=syncIdx:length(odometryData)
        if idx > length(thrustTime)
            break;
        end
        if odomTime(ii) >= thrustTime(idx)
            trajectory(idx,1:3) = (rot * (odometryData(ii, 1:3) - initInertial(1:3))')';
            trajectory(idx,4:6) = odometryData(ii,4:6);
            idx = idx + 1;
        else
            continue;
        end
    end

    % Convert to Marine Coordinate Frame
    trajectory(:,2:3) = trajectory(:,2:3) .* -1;
    trajectory(:,5:6) = trajectory(:,5:6) .* -1;
    
    %% Convert datetime to seconds
    time = seconds(thrustTime - thrustTime(1));
end