function phat=IC(y,m,option,pmax)
% ======================================================================
% Information Criterion for ADF test of unit root.
% Input:
%     y: test series in column vector
%     m: 0 with no constant and trend, 1 for constant only, 2 for both 
%        trend and constant 
%     option:  1: AIC, 2:BIC
%     pmax: maximum value of the lag
% Output:
%     phat: selected AR order
% =====================================================================
%  Yu Bai, April 23 2017
%  Note: AR(0) will not be selected !!
% =====================================================================

T=size(y,1);

% define the case if there are constant, trend or both
switch m
    case 1
        z=ones(T,1);
    case 2
        z=[ones(T,1),(1:T)'];
end

% case with deterministic trend only
yh=y(pmax+1:T,1);
if m>0
    z=z(pmax+1:T,:);
    uhat=yh-z*((z'*z)\(z'*yh));
    ee0=log(uhat'*uhat/size(yh,1));
end

% construct lag variables
Rg=zeros(T-pmax,pmax);
for idn=1:pmax
    Rg(:,idn)=y(pmax+1-idn:T-idn,1);
end

S=zeros(pmax,1);
for j=1:pmax
    switch option
        case 1      % AIC criterion
            kk=(2*j)/size(yh,1);
        case 2      % BIC criterion
            kk=log(size(yh,1))*j/size(yh,1);
    end

    if m>0
        X=[z,Rg(:,1:j)];
    else
        X=Rg(:,1:j);
    end
    
    bhat=(X'*X)\(X'*yh);
    logL=log((yh-X*bhat)'*(yh-X*bhat)/(size(yh,1)));
    S(j,1)=logL+kk;
end

if m>0
    S=[ee0;S];
    [~,phat]=min(S);
    if phat>1
       phat=phat-1;
    end
else 
    [~,phat]=min(S);
    if phat>1
       phat=phat-1;
    end
end