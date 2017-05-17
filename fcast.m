function yforst = fcast( y,option,pmax,h,t,method)
% =================================================================
%  Construct forecasts from AR forecasts
%  INPUT:
%     y: column time series vector for forecasting
%     option: 1 for AIC criterion, 2 for BIC criterion
%     pmax: set the maximum number of lags for selecting lag length
%     ilest: last estimation period
%     h: number of periods ahead for forecast (maximum)
%     t: start forecasting period
%     method: direct or iterated
%  OUTPUT:
%     yforst: forecasts of series y
% =====================================================================
%  Yu Bai,  April 25 2017
%  Notes: 1. We consider AR model  with a drift term 
%    2. Preliminary!! Comments are welcome.
% =====================================================================

% initial settings
m=1;
T=size(y,1);
yforst=zeros(h,1);

% lagged variables of dependent variable
Rg=zeros(T,pmax);
for idn=1:pmax
   Rg(idn+1:T,idn)=y(1:T-idn,1);
end

% AR forecasts
if strcmp(method,'direct')   % projection methods
inph=1;
while inph<=h
    % dependent variable
    ynew=y(1:t-inph);

    % determine the number of lags using AIC or BIC criterion
    phat=IC(ynew,m,option,pmax);
    
    xreg=[ones(t-inph,1) Rg(1:t-inph,1:phat)];  % regressors 
    xfor=[ones(1,1) Rg(t+1,1:phat)];  % forecasting
    
    % run the regression
    beta=(xreg'*xreg)\(xreg'*ynew);
    
    % construct AR forecasts
    yforst(inph)=xfor*beta;
    inph=inph+1;     
        
end
end

if strcmp(method,'iterated')   % projection methods
    yres=y(1:t);       % construct dependent variables
    phat=IC(yres,m,option,pmax);    % determine the number of lags using AIC or BIC criterion
    xreg=[ones(t,1) Rg(1:t,1:phat)];  % regressors 
    
 % run the regression
    beta=(xreg'*xreg)\(xreg'*yres);
    
    % compute dynamic forecast of y
    % construct companion form matrix
    if phat>1
        id=eye(plag);
        idr=id(1:plag-1,:);
        compm=[beta(m+1:end)';idr];
        const=[beta(m);idr(:,plag)];
    else
        compm=beta(m+1:end);
        const=beta(m);
    end
    
    hh=1;
    while hh<=h
        if hh==1
            compmp=compm;
            constmp=const;
        else
            constmp=constmp+compmp*constmp;
            compmp=compmp*compmp;
        end
        yforst(hh)=[ones(1,1) Rg(t+1,1:phat)]*[constmp(1,1) compmp(1,:)]';
        hh=hh+1;
    end
end
     
end