function dx= Protein_Detailed_Model(t,x,p,par)

dx = zeros(11,1);

%%%%molecule species

STAR = x(1);  
THS = x(2);       
TetR = x(3);
aTc = x(4);
aTc_TetR = x(5);   
Y = x(6);
Y_act = x(7);
Pz_rep = x(8); 
Pz_act = x(9);
Z = x(10);
GFP = x(11);


Pz = par.P_z - Pz_rep - Pz_act;

dx(1) = par.P_x*p(1) - p(12)*STAR*Pz - p(6)*STAR + p(13)*Pz_act; %(par.IPTG^p(20)/(p(21)^p(20)+par.IPTG^p(20))); % (par.IPTG^p(20));

dx(2) = par.P_x*p(1) - p(14)*THS*Y - p(7)*THS;

dx(3) = p(2)*Y_act - p(15)*TetR*Pz - p(16)*TetR*aTc - p(8)*TetR + p(17)*Pz_rep + p(18)*aTc_TetR - p(21)*TetR*Pz_act;

dx(4) = -p(16)*TetR*aTc  + p(18)*aTc_TetR;

dx(5) = p(16)*TetR*aTc  - p(18)*aTc_TetR;

dx(6) = p(3)*par.P_y - p(14)*THS*Y - p(9)*Y;

dx(7) = p(14)*THS*Y - p(20)*Y_act; 

dx(8) = p(15)*TetR*Pz - p(17)*Pz_rep + p(21)*TetR*Pz_act;

dx(9) = p(12)*STAR*Pz - p(13)*Pz_act - p(21)*TetR*Pz_act;

dx(10) = p(4)*Pz_act -  p(10)*Z; 

dx(11) = p(5)*Z - p(11)*GFP;


