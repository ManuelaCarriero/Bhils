%LOAD DATA
%data=zeros(9,4);
%data=data_1;
%%
%PCA


data_scored = zscore(data);

[coeff,score,latent,tsquared,explained,mu]=pca(data_scored);

%figure, scatter(coeff(:,1),coeff(:,2));
figure, scatter(score(:,1),score(:,2));

figure, bar(explained);
xlabel 'N Components'
ylabel 'Variance explained (%)'

figure, bar(coeff(:,1));
xlabel('Variable')
ylabel('Loading Scores')
ylim([0,1])

%params to choose
height=data(:,1);
n_obs=numel(height);
thr = 180;
x_comp=1;
y_comp=2;

%plot
figure,
for i = 1:n_obs
    if height(i,1) > thr
        scatter(score(i,x_comp),score(i,y_comp),350,'.','r')
        
        %PC labels  
        x_comp_str=num2str(x_comp);
        y_comp_str=num2str(y_comp);
        
        %var explained labels
        x_explained=round(explained(x_comp),2);
        y_explained=round(explained(y_comp),2);
        x_explained_str=num2str(x_explained);
        y_explained_str=num2str(y_explained);

        xlabel(strcat('PC',x_comp_str,'(',x_explained_str,'%)'));
        ylabel(strcat('PC',y_comp_str,'(',y_explained_str,'%)'));
        hold on
    else
        scatter(score(i,x_comp),score(i,y_comp),350,'.','b')
    end
end
%%
%PLS

[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(data_scored(:,2),data_scored(:,3:4));

figure
plot(1:2,cumsum(100*PCTVAR(2,:)),'-bo');
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in y');

figure,
scatter(XS(:,1),YS(:,1))

%params to choose
x_comp=1;
y_comp=1;

figure,
th = 180;
for i = 1:9
    if height(i,1) > th
        hold on
        scatter(XS(i,x_comp),YS(i,y_comp),350,'.','r')
        x_comp_str=num2str(x_comp);
        y_comp_str=num2str(y_comp);
        xlabel(strcat('PLS X',x_comp_str))
        ylabel(strcat('PLS Y',y_comp_str))

    else
        hold on
        scatter(XS(i,x_comp),YS(i,y_comp),350,'.','b')
        x_comp_str=num2str(x_comp);
        y_comp_str=num2str(y_comp);
        xlabel(strcat('PLS X',x_comp_str))
        ylabel(strcat('PLS Y',y_comp_str))
    end
end

figure, scatter(data(:,2),data(:,3))