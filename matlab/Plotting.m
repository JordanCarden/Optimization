clc
clear all

load circuit2_aTc50_cumultative.mat

With_TetR_2_AAV = AAV;
With_TetR_2_ASV = ASV;
With_TetR_2_LVA = LVA;
With_TetR_2_0 = noTetR;

tspan=0:0.1:420*60;  %%% seconds
options = odeset('RelTol',1e-11,'AbsTol',1e-11);


% total plasmid concentrations
par.P_z = 1e-9;
par.P_y = 1e-9;
par.P_x = 1e-9; 

atc_conv = 0.46822;
par.IPTG = 0.1*10^-3; %M


load fit_ASV.mat
p=p;
fit = p(8);
values_to_assign = [2*fit,fit,12*fit,0.0];
values_to_assign = round(values_to_assign, 5); % Rounds to 5 decimal places

for j = 1:4
    
    par.aTc = (50/atc_conv)*10^-9; %M
    x0 = [0 0 0 par.aTc 0 0 0 0 0 0 0];
    p(8) = values_to_assign(j);
    if j == 4 
        p(2) = 0;
    end
    [t,x] = ode23tb(@(t,x)Protein_Detailed_Model(t,x,p,par),tspan,x0);
    x = x.*10^9; %%% convert to nM for visualization
    Simout_TetR_j = x(1:6000:end,11).*10^p(19);
    tt = 70+(0:length(With_TetR_2_0)-1).*10; %%% for visualization
    if j == 1
        Simout_1 = Simout_TetR_j;
        figure(1)
        plot(tt,Simout_1,'-r',tt,With_TetR_2_AAV,'*r') 
        ylim([0 3000]);
        xlabel("Time (min)") 
        ylabel ("GFP") 
        hold on
    end
    if j == 2
        Simout_2 = Simout_TetR_j;
        figure(1)
        plot(tt,Simout_2,'-b',tt,With_TetR_2_ASV,'*b') 
        ylim([0 3000]); 
        xlabel("Time (min)") 
        ylabel ("GFP") 
        hold on
    end
    if j == 3
        Simout_3 = Simout_TetR_j;
        figure(1) 
        plot(tt,Simout_3,'-k',tt,With_TetR_2_LVA,'*k') 
        ylim([0 3000]);
        xlabel("Time (min)") 
        ylabel ("GFP") 
        hold on
    end
    if j == 4
        Simout_4 = Simout_TetR_j;
        figure(1)
        plot(tt,Simout_4,'-m',tt,With_TetR_2_0,'*m')
        ylim([0 3000]);
        xlabel("Time (min)")
        ylabel ("GFP")
        hold off
        legend ("AAV-sim","AAV-exp","ASV-sim","ASV-exp","LVA-sim","LVA-exp","noTetR-sim","noTetR-exp") 
  
    end
end


objective = sum((Simout_1 - With_TetR_2_AAV).^2)/max(With_TetR_2_AAV) + sum((Simout_2 - With_TetR_2_ASV).^2)/max(With_TetR_2_ASV) + ...
   sum((Simout_3 - With_TetR_2_LVA).^2)/max(With_TetR_2_LVA) + sum((Simout_2 - With_TetR_2_0).^2)/max(With_TetR_2_0);


disp(['SSE Objective: ' num2str(objective)])



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
