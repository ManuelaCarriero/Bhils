%LOAD DATA
%data=zeros(9,4);
%data=data_1;
%%
%data
%height, distance to work, weight, shoe size

data = load('data.txt');
data_scored = zscore(data);

data_size=size(data);
n_obs=data_size(1);
%%
%PCA

[coeff,score,latent,tsquared,explained,mu]=pca(data_scored);

figure, bar(explained);
xlabel 'N Components'
ylabel 'Variance explained (%)'

%choose which component to plot the 
% loading scores for
comp=1;

figure, bar(coeff(:,comp));
xlabel('Variable')
comp_str=num2str(comp);
ylabel(strcat('Loading Scores PC',comp_str))
ylim([0,1])

%params to choose
n_var=1; %variable which most accounts for variance
thr = 180;
x_comp=1;
y_comp=2;



%plot data in PCA latent space

figure,
for i = 1:n_obs
    if data(i,n_var) > thr
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
%height, distance to work, weight, shoe size


n_comp=3;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(data_scored(:,1:3),data_scored(:,4),n_comp);

%ncomp in PLS must be <= number of independent variables
%ncomp in Y is the same as the ncomp chosen

%plot variance explained by each component
%to understand which component to keep
%to have signal and discard noise 
% as much as possible
figure
plot(1:n_comp,cumsum(100*PCTVAR(2,:)),'-bo');
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in y');

figure
n_comp_str=num2str(n_comp);
bar(100*PCTVAR(1,1:n_comp))
xlabel(strcat('PLS X',n_comp_str));
ylabel('Percent of variance explained');

figure
bar(100*PCTVAR(2,1:n_comp))
xlabel(strcat('PLS Y',n_comp_str));
ylabel('Percent of variance explained');

%params to choose
%typically they are the same
x_comp=1;
y_comp=1;

figure,
th = 180;
for i = 1:n_obs
    if data(i,n_var) > th
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

%plot data to show the degree of correlation between variables
var1=1;
var2=2;
figure, scatter(data(:,var1),data(:,var2),350,'.','b');