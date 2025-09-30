location = "L:\My_drive\ME 578\project\project 1\automatic\";
odomFileName = "2025-09-16-14-17-31-odometry-navsat.csv";



[time, trajectory, thrusters] = GetRoboatRuntimeData(location, odomFileName);


plot(time,trajectory(:,1),'b',LineWidth=1.5)
hold on 
plot(time,trajectory(:,2),'g',LineWidth=1.5)
plot(time,trajectory(:,3),'r',LineWidth=1.5)


plot(time,trajectory(:,4),'b--',LineWidth=1.5)
hold on 
plot(time,trajectory(:,5),'g--',LineWidth=1.5)
plot(time,trajectory(:,6),'r--',LineWidth=1.5)


save('roboat_run.mat', 'thrusters', 'time', 'trajectory'); 

