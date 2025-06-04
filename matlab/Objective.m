function objective = Multiple_Obj_v2(p,par)


load circuit2_aTc50_cumultative

With_ASV = ASV;
With_noTetR = noTetR;


tspan=0:0.1:420*60;  %%% seconds / 660 => 480 needed
options = odeset('RelTol',1e-11,'AbsTol',1e-11); % 9 => 11 


% total plasmid concentrations
par.P_z = 1e-9;
par.P_y = 1e-9;
par.P_x = 1e-9; 

par.IPTG = 0.1*10^-3; %M
atc_conv = 0.46822;


save fit.mat p
% load fit_ASV.mat
% p=p;

 for j = 1:2
   
    new_p = p; % new parameter for RBP = 0 
    
    if j == 1
        par.aTc = (50/atc_conv)*10^-9; %M
        x0 = [0 0 0 par.aTc 0 0 0 0 0 0 0];
        [t,x] = ode23tb(@(t,x)Protein_Detailed_Model(t,x,p,par),tspan,x0);
        x = x.*10^9; %%% convert to nM for visualization
        Simout_RBP = x(1:6000:end,11).*10^p(19);
        tt = 0:length(With_noTetR)-1; %%% for visualization  
        Simout_1 = Simout_RBP;
        figure(1)
        plot(tt,Simout_1,'-b',tt,With_ASV,'*b') 
        ylim([0 2800]);
        xlabel("Time (min)") 
        ylabel ("GFP")
        hold on
    end
    if j == 2 
        par.aTc = (50/atc_conv)*10^-9; %M
        x0 = [0 0 0 par.aTc 0 0 0 0 0 0 0];
        new_p(2) = 0;
        new_p(8) = 0;
        [t,x] = ode23tb(@(t,x)Protein_Detailed_Model(t,x,new_p,par),tspan,x0); %new_p
        x = x.*10^9; %%% convert to nM for visualization
        Simout_RBP = x(1:6000:end,11).*10^p(19);
        tt = 0:length(With_noTetR)-1; %%% for visualization
        Simout_2 = Simout_RBP;
        figure(1)
        plot(tt,Simout_2,'-m',tt,With_noTetR,'*m') 
        ylim([0 2800]);
        xlabel("Time (min)") 
        ylabel ("GFP") 
        hold off
        legend("ASV-sim","ASV-exp","noTetR-sim","noTetR-exp")
    end
   
 end

% save newparameter_noTetR new_p

objective = sum((Simout_1 - With_ASV).^2)/max(With_ASV) + sum((Simout_2 - With_noTetR).^2)/max(With_noTetR);
% objective = sum((Simout_2 - With_noTetR).^2)/max(With_noTetR);

disp(['SSE Objective: ' num2str(objective)])
disp(['Simu1_End: ' num2str(Simout_1(end)),  'Exp1: ' num2str(With_ASV(end))])
disp(['Simu2_End: ' num2str(Simout_2(end)),  'Exp2: ' num2str(With_noTetR(end))])

figure(2)
names = ["STAR","THS","TetR","aTc","aTc:TetR","Y","Yact","Pzrep","Pzact","Z","GFP" ];

Pz = par.P_z*10^(9) - x(:,8) - x(:,9); 

for i = 1:11
    subplot(3,4,i)
    plot(x(1:6000:end,i),'LineWidth',2)
    title(names(i),'HorizontalAlignment','left')
    xlabel("Time (min)")
    set(gca,'FontSize',18)
    set(gca,'FontName','Times New Roman')
end

subplot(3,4,12)
plot(Pz(1:6000:end),'LineWidth',2)
title('Pzfree','HorizontalAlignment','left')
xlabel("Time (min)")
set(gca,'FontSize',18)
set(gca,'FontName','Times New Roman')
