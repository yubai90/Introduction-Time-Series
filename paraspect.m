function [spectrum,freqs] = paraspect(Y)
% ======================================================================
%  Parametric estiation of spectrum 
%  INPUT:
%      y: time series used for estimation of the spectrum
%      lag: maximum number of estimated covariance
%  OUTPUT:
%      spectrum: estimation results of the spectrum
%      freqs: frequencies
% =======================================================================
%   Yu Bai, April 30 2017
% =======================================================================

% estimate the model, get the required parameters
[alpha,beta,~,~,sigma,p,q] = arma(Y,0);   % assume no constant term

nfreq=100;
freqs=linspace(0,pi,nfreq);

b1=zeros(1,nfreq);
b2=zeros(1,nfreq);
a1=zeros(1,nfreq);
a2=zeros(1,nfreq);

jj=1;
while jj<=q
    b1b=beta(jj)*exp(1i*(-jj*freqs));
    b2b=beta(jj)*exp(1i*jj*freqs);
    b1=b1+b1b;
    b2=b2+b2b;
    jj=jj+1;
end

qq=1;
while qq<=p
    a1a=alpha(qq)*exp(1i*(-qq*freqs));
    a2a=alpha(qq)*exp(1i*qq*freqs);
    a1=a1+a1a;
    a2=a2+a2a;
    qq=qq+1;
end

spectrum=sigma/(2*pi)*((1+b1).*(1+b2))./((1-a1).*(1-a2));

function [alpha,beta,c,se,sigma,p,q] = arma(Y,m)
% ======================================================================
%  Estimate ARMA(p,q) model using conditional MLE
%  [theta,se,logl,res] = armamle(X,AR,MA,C)
%  INPUT:
%      Y: observations
%      m: m=0 if there is no constant term; m=1 if there is a drift term; m=2
%        if there is a linear trend terml
%  OUTPUT:
%      theta: estimated parameters
%      se: standard error
%      sigma: estimated sigma^2
% ======================================================================
%  Notes: code is modified version from m file which is available 
%     from Prof. Junhui qian's website(plus lag length selection).
% ======================================================================

% determine the lag length of AR, run OLS regression and get the residuals
T=size(Y,1);
p=IC(Y,m,2,8);     % using BIC criterion, maximum lags is 8

% deterministic term
switch m
    case 1
        z=ones(T,1);
    case 2
        z=[ones(T,1),(1:T)'];
end

% construct the lag variables of Y
Rg=zeros(T-p,p);
for idn=1:p
    Rg(:,idn)=Y(p+1-idn:T-idn,1);
end
if m>0
    X=[z(p+1:T,:) Rg];
else
    X=Rg;
end

% OLS regression for the residuals
Ynew=Y(p+1:T,1);
beta0=(X'*X)\(X'*Ynew);
res=Ynew-X*beta0;

q=IC(res,0,2,8);  % determine lag length of MA, assume no deterministic term, use BIC,maximum number of lags is 8

total = p + q + m + 1;    % number of parameters.
theta0 = [zeros(total-1,1);std(Y)];
options = optimoptions(@fminunc,'Algorithm','quasi-newton');
options.MaxIter = 2000;
options.Display='off';
[theta,~,~,~,~,hessian] =fminunc(@lharma,theta0,options,Ynew,p,q,m);
V = inv(hessian);
se = sqrt(diag(V));

% estimate residual
if p>0
    alpha = theta(1:p);
end
if q>0
    beta = theta(p+1:p+q);
end
if m==1
    c = theta(1+p+q);
    const=c*ones(T,1);
elseif m==2
    c = theta(1+p+q:end-1);
    const=[ones(T,1) (1:T)']*c;
else
    c = 0;
    const = c;
end
e = zeros(T,1);
mpq = max(p,q);
if p > 0 && q > 0 
    for ii = mpq+1:T
        e(ii) = Y(ii) - const - alpha'*Y(ii-1:-1:ii-p) ...
        - beta'*e(ii-1:-1:ii-q);
    end
end
if p > 0 && q == 0
    for ii = mpq+1:T
        e(ii) = Y(ii) - const - alpha'*Y(ii-1:-1:ii-p); 
    end
end
if p==0 && q > 0
    for ii = mpq+1:T
        e(ii) = Y(ii) - const - beta'*e(ii-1:-1:ii-q);
    end
end
    
sigma = (1/(T-mpq))*(e'*e)';

function y=lharma(theta0,Y,p,q,m)
% likelihood function of ARMA(p,q) model

total = p + q + m + 1; % number of parameters.

T = length(Y);

A = zeros(p,1);
B = zeros(q,1);
if p>0
    alpha = theta0(1:p);
    A = alpha;
end
if q>0
    beta = theta0(p+1:p+q);
    B = beta;
end
if m==1
    c = theta0(1+p+q);
    const=c*ones(T,1);
elseif m==2
    c = theta0(1+p+q:end-1);
    const=[ones(T,1) (1:T)']*c;
else
    const = 0;
end
sigma = theta0(total);

y = 0;
e = zeros(T,1);
mpq = max(p,q);
if p>0 && q>0
    for ii = mpq+1:T
        e(ii) = Y(ii) - const - A'*Y(ii-1:-1:ii-p) ...
            - B'*e(ii-1:-1:ii-q);
        y = y + e(ii)^2;
    end
end
if p>0 && q==0
    for ii = mpq+1:T
        e(ii) = Y(ii) - const - A'*Y(ii-1:-1:ii-p);
        y = y + e(ii)^2;
    end
end
if p==0 && q>0
    for ii = mpq+1:T
        e(ii) = Y(ii) - const - B'*e(ii-1:-1:ii-q);
        y = y + e(ii)^2;
    end
end

y = y/(2*sigma^2);
y = y + log(sigma^2)*(T-mpq)/2 + log(2*pi)*(T-mpq)/2;

if isnan(y) || isinf(y) || ~isreal(y)
    y = 1e10;
end
