function MakeScurveData(t)
    tt = [-1:0.1:0.5]*pi;
    uu = tt(end:-1:1);
    hh = [0:0.1:1]*5;
    xx = [cos(tt) -cos(uu)]'*ones(size(hh));
    yy = ones(size([tt uu]))'*hh;
    zz = [sin(tt) 2-sin(uu)]'*ones(size(hh));
    cc = [tt uu]' * ones(size(hh));
    xx = xx + t*randn(size(xx));
    yy = yy + t*randn(size(yy));
    zz = zz + t*randn(size(zz));
    data3 = [xx(:),yy(:),zz(:)];
    figure;
    hold on;
    p = plot3(xx(:),yy(:),zz(:),'.','Markersize',20);daspect([1,1,1]);
    set(gca,'fontsize',16);view(3); 
    
    writematrix(data3, 'ScurveDataNoisy.csv');
end