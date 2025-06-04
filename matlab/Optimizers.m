clear all
clc

% total plasmid concentrations  everything in Molar
par.P_z = 1e-9;
par.P_y = 1e-9;
par.P_x = 1e-9; 

A = [];
b = [];
Aeq = [];
beq = [];
nlcon = [];


% new
lb = [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0001, 0.0001, 0.00005, 0.0001, 0.0001, 0.00005, 100, 0.00000001, 100, 100, 100, 0.00000001, 0.00000001, 0.5,0.0001,100];
ub = [0.5, 0.5, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1000000, 0.00001, 1000000, 1000000, 1000000, 0.00001, 0.00001, 5,0.01,1000000];


% 
% Evalall = [];
% min_difference = 0.002;
% for i = 1:length(lb)
%     if (ub(i) - lb(i)) < min_difference
%         ub(i) = lb(i) + min_difference;
%     end
% end


load initial_p.mat
p0=p;


p = fmincon(@Multiple_Obj_v2,p0,A,b,Aeq,beq,lb,ub,nlcon);
% p = simulannealbnd(@Multiple_Obj_v2, p0, lb, ub);
