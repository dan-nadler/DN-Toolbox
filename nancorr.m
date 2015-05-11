function r = nancorr(x,y);

% nancorr returns the correlation between two variables, even if they have 
%    different starting and ending dates. 
% x and y both column vectors. They must have the same length with nan's 
%    representing observations where data is unavailable.
% r is the correlation between x and y
%
% Schoen, 2000 :P

if ~isnan(nansum(x)) &  ~isnan(nansum(y)) 

if nargin == 2
  X = x(~isnan(x(:,1)) & ~isnan(y(:,1)));  
  Y = y(~isnan(x(:,1)) & ~isnan(y(:,1)));    
  if nancount(X)<=1 | nancount(Y)<=1;
      r = NaN;
  else
  r = corr(X,Y); 
  end;
else
  X = x(~isnan(x(:,1)) & ~isnan(x(:,2)));
  r = corrcoef(X);  
end    

else 
    
  r = nan;  

end

