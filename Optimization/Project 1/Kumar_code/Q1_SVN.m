function Q12() 
close all
%% read data
A2012 = readmatrix('A2012.csv');
A2016 = readmatrix('A2016.csv');
% Format for A2012 and A2016:
% FIPS, County, #DEM, #GOP, then <str> up to Unemployment Rate
str = ["Median Income", "Migration Rate", "Birth Rate",...
"Death Rate", "Bachelor Rate", "Unemployment Rate","log(#Votes)"];
%
% remove column county that is read by matlab as NaN
A2012(:,2) = [];
A2016(:,2) = [];
%% Remove rows with missing data
A = A2016;
% remove all rows with missing data
ind = find(~isfinite(A(:,2)) |  ~isfinite(A(:,3)) | ~isfinite(A(:,4)) ...
    | ~isfinite(A(:,5)) | ~isfinite(A(:,6)) | ~isfinite(A(:,7)) ...
    | ~isfinite(A(:,8)) | ~isfinite(A(:,9)));
A(ind,:) = [];
%% select CA, OR, WA, NJ, NY counties
 ind = find((A(:,1)>=6000 & A(:,1)<=6999)); % ...  %CA
%  | (A(:,1)>=53000 & A(:,1)<=53999) ...        %WA
%  | (A(:,1)>=34000 & A(:,1)<=34999) ...        %NJ  
%  | (A(:,1)>=36000 & A(:,1)<=36999) ...        %NY
%  | (A(:,1)>=41000 & A(:,1)<=41999));          %OR
 A = A(ind,:);

[n,dim] = size(A)

%% assign labels: -1 = dem, 1 = GOP
idem = find(A(:,2) >= A(:,3));
igop = find(A(:,2) < A(:,3));
num = A(:,2)+A(:,3);
label = zeros(n,1);
label(idem) = -1;
label(igop) = 1;

%% select max subset of data with equal numbers of dem and gop counties
% ngop = length(igop);
% ndem = length(idem);
% if ngop > ndem
%     rgop = randperm(ngop,ndem);
%     Adem = A(idem,:);
%     Agop = A(igop(rgop),:);
%     A = [Adem;Agop];
% else
%     rdem = randperm(ndem,ngop);
%     Agop = A(igop,:);
%     Adem = A(idem(rdem),:);
%     A = [Adem;Agop];
% end  
% [n,dim] = size(A)
% idem = find(A(:,2) >= A(:,3));
% igop = find(A(:,2) < A(:,3));
% num = A(:,2)+A(:,3);
% label = zeros(n,1);
% label(idem) = -1;
% label(igop) = 1;

%% set up data matrix and visualize
close all
figure;
hold on; grid;
X = [A(:,4:9),log(num)];
X(:,1) = X(:,1)/1e4;
% select three data types that distinguish dem and gop counties the most
i1 = 1; % Median Income
i2 = 7; % log(# votes)
i3 = 5; % Bachelor Rate
plot3(X(idem,i1),X(idem,i2),X(idem,i3),'.','color','b','Markersize',20);
plot3(X(igop,i1),X(igop,i2),X(igop,i3),'.','color','r','Markersize',20);
view(3)
fsz = 16;
set(gca,'Fontsize',fsz);
xlabel(str(i1),'Fontsize',fsz);
ylabel(str(i2),'Fontsize',fsz);
zlabel(str(i3),'Fontsize',fsz);
%% rescale data to [0,1] and visualize
figure;
hold on; grid;
XX = X(:,[i1,i2,i3]); % data matrix
% rescale all data to [0,1]
xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
X1 = (XX(:,1)-xmin)/(xmax-xmin);
X2 = (XX(:,2)-ymin)/(ymax-ymin);
X3 = (XX(:,3)-zmin)/(zmax-zmin);
XX = [X1,X2,X3];
plot3(XX(idem,1),XX(idem,2),XX(idem,3),'.','color','b','Markersize',20);
plot3(XX(igop,1),XX(igop,2),XX(igop,3),'.','color','r','Markersize',20);
view(3)
fsz = 16;
set(gca,'Fontsize',fsz);
xlabel(str(i1),'Fontsize',fsz);
ylabel(str(i2),'Fontsize',fsz);
zlabel(str(i3),'Fontsize',fsz);
%% set up optimization problem
[n,dim] = size(XX);
G=zeros(dim+n+1,dim+n+1);
Gtemp=eye(dim+1,dim+1);
G(1:dim,1:dim)=eye(dim,dim);
y=label;
Atemp=y*ones(1,dim+1).*[XX,ones(n,1)]
A=eye(2*n,dim+n+1);
A(1:n,1:dim+1)=Atemp;
A(n+1:2*n,1:dim+1)=zeros(n,dim+1);

btemp=ones(n,1);
b=[ones(n,1);zeros(n,1)];
w = ones(dim+1+n,1);
wtemp = ones(dim+1,1);
C=10000;
fun = @(w)fun0(w,G,C);
gfun = @(w)gfun0(w,G,C);
Hvec = @()Hvec0(w,G,C);

%[w,f,gnorm] = SINewton(fun,gfun,Hvec,Y,w);
%Guess the initial values 
[wtemp,l,lcomp] = FindInitGuess(wtemp,Atemp,btemp);
w=[wtemp;lcomp];
size(w)
W=[];
fprintf('prev w = [%d,%d,%d], b = %d\n',w(1),w(2),w(3),w(4));
%Use the ASM function to compute plane parameters
[w,lm] = ASM(w,gfun,Hvec,A,b,W);
size(w)
fprintf('w = [%d,%d,%d], b = %d\n',w(1),w(2),w(3),w(4));
lcomp=lcomp(:,1);
xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
nn = 50;
[xx,yy,zz] = meshgrid(linspace(xmin,xmax,nn),linspace(ymin,ymax,nn),...
    linspace(zmin,zmax,nn));
plane = w(1)*xx+w(2)*yy+w(3)*zz+w(4);
p = patch(isosurface(xx,yy,zz,plane,0));
p.FaceColor = 'green';
p.EdgeColor = 'none';
camlight 
lighting gouraud
alpha(0.3);

%%
% figure;
% hold on;
% grid;
% niter = length(f);
% plot((0:niter-1)',f,'Linewidth',2);
% set(gca,'Fontsize',fsz);
% xlabel('k','Fontsize',fsz);
% ylabel('f','Fontsize',fsz);
%%
% figure;
% hold on;
% grid;
% niter = length(gnorm);
% plot((0:niter-1)',gnorm,'Linewidth',2);
% set(gca,'Fontsize',fsz);
% set(gca,'YScale','log');
% xlabel('k','Fontsize',fsz);
% ylabel('|| stoch grad f||','Fontsize',fsz);

end
%%
function f = fun0(w,G,C)
f = 0.5*w'*G*w+C*[zeroes(1,dim+1),ones(1,n)]*w;
end
%%
function g = gfun0(w,G,C)
g = G*w;
end
%%
function Hv = Hvec0(w,G,C)
Hv=G;
end