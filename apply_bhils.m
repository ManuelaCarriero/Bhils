%% parameters to be chosen by the user
% if you want a monovariate PLS or multivariate PLS toydataset latent space

pls_type = 'monovariate_y';%'monovariate_x'; %monovariate_y %multivariate
variable = 'shoe size'; %if monovariate_x: height, distance to work %if monovariate_y: weight, shoe size

% number of components for the clustering of the EMI dataset in the PCA
% latent space
n_comp_for_clustering = 2; % we explored: %3%4%5&6

%% toy dataset analysis

%% load data
disp('load toydataset')
%variables
%height, distance to work, weight, shoe size

data = load('data.txt');
data_scored = zscore(data);

data_size=size(data);
n_obs=data_size(1);

%% check the correlation between variables
disp('calculate correlation between variables')

%are independent variables correlated ?
[r12,p12] = corrcoef(data_scored(:,1),data_scored(:,2));
%are dependent variables correlated ?
[r34,p34] = corrcoef(data_scored(:,3),data_scored(:,4));
%correlation between Y1 and X1 and X2
[r31,p31] = corrcoef(data_scored(:,3),data_scored(:,1));
[r32,p32] = corrcoef(data_scored(:,3),data_scored(:,2));
%correlation between Y2 and X1 and X2
[r41,p41] = corrcoef(data_scored(:,4),data_scored(:,1));
[r42,p42] = corrcoef(data_scored(:,4),data_scored(:,2));

%% PCA
disp('apply PCA to the toydataset')

[coeff,score,latent,tsquared,explained,mu]=pca(data_scored);

%%
disp('PLOT1 and PLOT2: understand the variable which explains most the variance using PCA scree plot')

figure, bar(explained);
set(get(gca, 'XAxis'), 'FontWeight', 'bold','FontSize',14);
set(get(gca, 'YAxis'), 'FontWeight', 'bold','FontSize',14);
set(gca,'box','off')
xlabel 'N Components'
ylabel 'Variance explained (%)'
title 'PLOT1 - PCA'
grid on

%choose which component to plot the 
% loading scores for
comp=1;

figure, bar(coeff(:,comp));
xticklabels({'Height','Distance','Weight','Shoe Size'})
set(get(gca, 'XAxis'), 'FontWeight', 'bold','FontSize',14);
set(get(gca, 'YAxis'), 'FontWeight', 'bold','FontSize',14);
set(gca,'box','off')
xlabel('Variable')
comp_str=num2str(comp);
ylabel(strcat('Loading Scores PC',comp_str))
ylim([0,1])
title 'PLOT 2 - PCA'
grid on

%%
disp('choose threshold for the color of samples based on the height variable')
%params to choose
n_var=1; %variable which most accounts for variance
thr = 180;
x_comp=1;
y_comp=2;

%%
disp('PLOT 3 - toydataset in PLS latent space')

%plot data in PCA latent space

figure,
for i = 1:n_obs
    if data(i,n_var) > thr %put colors based on the height value of the sample
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
    set(get(gca, 'XAxis'), 'FontWeight', 'bold','FontSize',14);
    set(get(gca, 'YAxis'), 'FontWeight', 'bold','FontSize',14);
    set(gca,'box','off')
    title 'PLOT 3 - TOYDATASET in PCA latent space'
    grid on
end

%% PLS
%height, distance to work, weight, shoe size


%n_components = maximum number of independent variables X.
if strcmp(pls_type,'multivariate')
    x_variables = data_scored(:,1:2);
    y_variables = data_scored(:,3:4);
    title_pls='PLOT 6 - TOYDATASET in PLS latent space - Multivariate X and Y';
elseif strcmp(pls_type,'monovariate_x') && strcmp(variable,'height')
    x_variables = data_scored(:,1);
    y_variables = data_scored(:,3:4);
    title_pls='PLOT 6 - TOYDATASET in PLS latent space - Monovariate X (height)';
elseif strcmp(pls_type,'monovariate_x') && strcmp(variable,'distance to work')
    x_variables = data_scored(:,2);
    y_variables = data_scored(:,3:4);
    title_pls='PLOT 6 - TOYDATASET in PLS latent space - Monovariate X (distance to work)';
elseif strcmp(pls_type,'monovariate_y') && strcmp(variable,'weight')
    x_variables = data_scored(:,1:2);
    y_variables = data_scored(:,3);
    title_pls='PLOT 6 - TOYDATASET in PLS latent space - Monovariate Y (weight)';
elseif strcmp(pls_type,'monovariate_y') && strcmp(variable,'shoe size')
    x_variables = data_scored(:,1:2);
    y_variables = data_scored(:,4);
    title_pls='PLOT 6 - TOYDATASET in PLS latent space - Monovariate Y (shoe size)';
end

n_comp=length(x_variables(1,:));
n_var=1; %variable which most accounts for variance (height) as seen from PCA scree plot

%%

[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(x_variables,y_variables,n_comp);%data_scored(:,1:2)

%ncomp in PLS must be <= number of independent variables

%%
%plot variance explained by each component
%to understand which component to keep
%to have signal and discard noise 
% as much as possible

% figure
% plot(1:n_comp,cumsum(100*PCTVAR(1,:)),'-bo');
% xlabel('Number of PLS components');
% ylabel('Percent Variance Explained in x');
% 
% figure
% plot(1:n_comp,cumsum(100*PCTVAR(2,:)),'-bo');
% xlabel('Number of PLS components');
% ylabel('Percent Variance Explained in y');
disp('PLOT 4 and 5 - understand the variable which explains most the variance using weights of PLS components')

figure,
x_pctvar=1;
bar(100*PCTVAR(x_pctvar,1:n_comp))
xticklabels({'Height','Distance'})
set(get(gca, 'XAxis'), 'FontWeight', 'bold','FontSize',14);
set(get(gca, 'YAxis'), 'FontWeight', 'bold','FontSize',14);
set(gca,'box','off')
xlabel('PLS X');
ylabel('Loading values');
title 'PLOT 4 - PLS'
grid on

figure,
y_pctvar=2;
bar(100*PCTVAR(y_pctvar,1:n_comp));
set(get(gca, 'XAxis'), 'FontWeight', 'bold','FontSize',14);
set(get(gca, 'YAxis'), 'FontWeight', 'bold','FontSize',14);
set(gca,'box','off')
xlabel('PLS Y');
ylabel('Loading values');
title 'PLOT 5 - PLS'
grid on

%%
disp('PLOT 6 - toydataset in PLS latent space')

%params to choose
%typically they are the same
%in this case we choose both for x and y
%the first component 
%which mostly explains the variance in both cases (i.e. along x and along y).
x_comp=1;
y_comp=1;

figure,
th = 180;
for i = 1:n_obs
    if data(i,n_var) > th %put colors based on the height value of the sample
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
    set(get(gca, 'XAxis'), 'FontWeight', 'bold','FontSize',14);
    set(get(gca, 'YAxis'), 'FontWeight', 'bold','FontSize',14);
    set(gca,'box','off')
    grid on
    title(title_pls);
end

%% EMI data latent space and unsupervised clustering
data_emi=load('EMI_ROIs.mat');
data_emi=cat(2,data_emi.y,data_emi.x_rsoma,data_emi.x_fsoma,data_emi.x_fneurite,data_emi.x_fextra,data_emi.x_De,data_emi.x_Din);
cmro2 = data_emi(:,1);
rsoma = data_emi(:,2);
fsoma = data_emi(:,3);
fneurite = data_emi(:,4);
fextra = data_emi(:,5);
De = data_emi(:,6);
Din = data_emi(:,7);

%% PCA 



% %plot data to show the degree of correlation between variables
% var1=1;
% var2=2;
% figure, scatter(data(:,var1),data(:,var2),350,'.','b');

data_emi_scored=zscore(data_emi);
[coeff,score,latent,tsquared,explained,mu]=pca(data_emi_scored);

%%
disp('PLOT 7 - 8: PCA - EMI data, check the scree plot and loading scores')

fig=figure; 
bar(explained);
set(get(gca, 'XAxis'), 'FontWeight', 'bold','FontSize',14);
set(get(gca, 'YAxis'), 'FontWeight', 'bold','FontSize',14);
set(gca,'box','off')
xlabel('N Components','FontWeight','bold')
ylabel('Variance explained (%)','FontWeight','bold')
%title 'PLOT 7 - PCA EMI dataset'
grid on
saveas(fig,'PCA_PCsVariance_EMIdataset.svg')
%choose which component to plot the 
% loading scores for
comp=1;

fig=figure; 
bar(coeff(:,comp));
xticklabels({'CMRO_2','R_{soma}','f_{soma}','f_{neurite}','f_{extra}','D_e','D_{in}'})
set(get(gca, 'XAxis'), 'FontWeight', 'bold','FontSize',14);
set(get(gca, 'YAxis'), 'FontWeight', 'bold','FontSize',14);
set(gca,'box','off')
xlabel('Variable','FontWeight','bold')
comp_str=num2str(comp);
ylabel(strcat('Loading Scores PC',comp_str),'FontWeight','bold')
%ylim([0,1])
%title 'PLOT 8 - PCA EMI dataset'
grid on
saveas(fig,'PCA_loadingscoresPC1_EMIdataset.svg')

%%
disp('PLOT 9 - plot EMI dataset in PCA latent space')
%clustering
epsilon=1;
minpts=5;

%CHOOSE!!!!!!!!!!
%choose the number of components to be used for the dbscan clustering
if n_comp_for_clustering == 2
    pca_components_for_clustering = score(:,1:2);
elseif n_comp_for_clustering == 3
    pca_components_for_clustering = score(:,1:3);
elseif n_comp_for_clustering == 4
    pca_components_for_clustering = score(:,1:4);
elseif n_comp_for_clustering == 5
    pca_components_for_clustering = score(:,1:5);
elseif n_comp_for_clustering == 6
    pca_components_for_clustering = score(:,1:6);
end

idx = dbscan(pca_components_for_clustering,epsilon,minpts);

%check the class assigned to each sample
unique(idx);

x_comp=1;
y_comp=2;


%save samples values which are labelled with black colour
cmro2_value_cluster_pca = [];
rsoma_value_cluster_pca = [];
fsoma_value_cluster_pca = [];
fneurite_value_cluster_pca = [];
fextra_value_cluster_pca = [];
De_value_cluster_pca = [];
Din_value_cluster_pca = [];

fig=figure;
for i = 1:length(idx)
    if idx(i)==1
        scatter(score(i,1), score(i,2), 350, '.', 'm');
    hold on
    elseif idx(i)==-1
        scatter(score(i,1), score(i,2), 350, '.', 'k');
        cmro2_value_cluster_pca(end+1) = cmro2(i);
        rsoma_value_cluster_pca(end+1) = rsoma(i);
        fsoma_value_cluster_pca(end+1) = fsoma(i);
        fneurite_value_cluster_pca(end+1) = fneurite(i);
        fextra_value_cluster_pca(end+1) = fextra(i);
        De_value_cluster_pca(end+1) = De(i);
        Din_value_cluster_pca(end+1) = Din(i);
    hold on
    elseif idx(i)>1
        scatter(score(i,1), score(i,2), 350, '.', 'c');
    end
    set(get(gca, 'XAxis'), 'FontWeight', 'bold','FontSize',14);
    set(get(gca, 'YAxis'), 'FontWeight', 'bold','FontSize',14);
    set(gca,'box','off')
    grid on
    %var explained labels
    x_explained=round(explained(x_comp),2);
    y_explained=round(explained(y_comp),2);
    x_explained_str=num2str(x_explained);
    y_explained_str=num2str(y_explained);

    xlabel(strcat('PC',num2str(x_comp),'(',x_explained_str,'%)'),'FontWeight','bold');
    ylabel(strcat('PC',num2str(y_comp),'(',y_explained_str,'%)'),'FontWeight','bold');
    %title(strcat('PLOT 9 - EMI data, PCA latent space, N\_comp\_for\_clustering=',num2str(n_comp_for_clustering),' PCs'))
end
saveas(fig,strcat('PCA_latentspace_n',num2str(n_comp_for_clustering),'comp_EMIdataset.svg'))

%% a check
% figure,
% scatter(score(:,1),score(:,2))
% 
% %original space
% figure,
% data_emi=load('EMI_ROIs.mat');
% scatter(data_emi.x_rsoma,data_emi.y)


%% PLS

x_emi = cat(2,data_emi_scored(:,2),data_emi_scored(:,3),data_emi_scored(:,4),data_emi_scored(:,5),data_emi_scored(:,6),data_emi_scored(:,7));
n_comp=length(x_emi(1,:));
y_emi = data_emi_scored(:,1);

[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(x_emi,y_emi,n_comp);


%%
disp('PLOT 10 - 11: PLS - EMI data, check the scree plot and loading scores')
%COMPONENTS ARE COLUMNS OF XL
fig=figure; 
bar(XL(:,1));
xticklabels({'R_{soma}','f_{soma}','f_{neurite}','f_{extra}','D_e','D_{in}'})
set(get(gca, 'XAxis'), 'FontWeight', 'bold','FontSize',14);
set(get(gca, 'YAxis'), 'FontWeight', 'bold','FontSize',14);
set(gca,'box','off')
xlabel('X loadings (First Component)','FontWeight','bold')
ylabel('Loadings values','FontWeight','bold')
%title('PLOT 10 - PLS EMI dataset','FontWeight','bold')
grid on
saveas(fig,'PLS_Xloadings_EMIdataset.svg')

%choose which component to plot the 
% loading scores for
comp=1;

fig=figure; 
bar(YL);
xticklabels({'R_{soma}','f_{soma}','f_{neurite}','f_{extra}','D_e','D_{in}'})
set(get(gca, 'XAxis'), 'FontWeight', 'bold','FontSize',14);
set(get(gca, 'YAxis'), 'FontWeight', 'bold','FontSize',14);
set(gca,'box','off')
xlabel('Y loadings','FontWeight','bold')
comp_str=num2str(comp);
ylabel('Loadings values','FontWeight','bold')
%ylim([0,1])
%title('PLOT 11 - PLS EMI dataset','FontWeight','bold')
grid on
saveas(fig,'PLS_loadingvalues_EMIdataset.svg')

%%
disp('PLOT 12 - plot EMI dataset in PLS latent space')

%clustering
epsilon=1;
minpts=5;
data_clustering=cat(2,XS(:,1),YS(:,1));
idx = dbscan(data_clustering,epsilon,minpts);
unique(idx);

%save samples values which are labelled with black colour
cmro2_value_cluster_pls = [];
rsoma_value_cluster_pls = [];
fsoma_value_cluster_pls = [];
fneurite_value_cluster_pls = [];
fextra_value_cluster_pls = [];
De_value_cluster_pls = [];
Din_value_cluster_pls = [];

fig=figure;
for i = 1:length(idx)
    if idx(i)>0
        scatter(XS(i,1), YS(i,1), 350, '.', 'm');
    hold on
    else
        scatter(XS(i,1), YS(i,1), 350, '.', 'k');
        cmro2_value_cluster_pls(end+1) = cmro2(i);
        rsoma_value_cluster_pls(end+1) = rsoma(i);
        fsoma_value_cluster_pls(end+1) = fsoma(i);
        fneurite_value_cluster_pls(end+1) = fneurite(i);
        fextra_value_cluster_pls(end+1) = fextra(i);
        De_value_cluster_pls(end+1) = De(i);
        Din_value_cluster_pls(end+1) = Din(i);
    hold on
    end
    set(get(gca, 'XAxis'), 'FontWeight', 'bold','FontSize',14);
    set(get(gca, 'YAxis'), 'FontWeight', 'bold','FontSize',14);
    set(gca,'box','off')
    grid on
    xlabel('PLS X1','FontWeight','bold')
    ylabel('PLS Y1','FontWeight','bold')
    %title('PLOT 10 - EMI data, PLS latent space','FontWeight','bold')
end

saveas(fig,'PLS_latentspace_EMIdataset.svg')

%% check where these samples are


fig=figure;
hist(cmro2);
hold on
plot(cmro2_value_cluster_pca,zeros(1,length(cmro2_value_cluster_pca)),'o','MarkerSize',10,'MarkerFaceColor','red');
hold on
plot(cmro2_value_cluster_pls,zeros(1,length(cmro2_value_cluster_pls)),'o','MarkerSize',10,'MarkerFaceColor','cyan');
xlabel('CMRO_2 regional values','FontSize',14,'FontWeight','bold')
ylabel('counts','FontSize',14,'FontWeight','bold')
legend('','PCA cluster','PLS cluster')
ylim([0,15]);
grid on
saveas(fig,'dist_cmro2.svg')

fig=figure;
hist(rsoma);
hold on
plot(rsoma_value_cluster_pca,zeros(1,length(rsoma_value_cluster_pca)),'o','MarkerSize',10,'MarkerFaceColor','red');
hold on
plot(rsoma_value_cluster_pls,zeros(1,length(rsoma_value_cluster_pls)),'o','MarkerSize',10,'MarkerFaceColor','cyan');
xlabel('R_{soma} regional values','FontSize',14,'FontWeight','bold')
ylabel('counts','FontSize',14,'FontWeight','bold')
legend('','PCA cluster','PLS cluster')
grid on
saveas(fig,'dist_Rsoma.svg')

fig=figure;
hist(fsoma);
hold on
plot(fsoma_value_cluster_pca,zeros(1,length(fsoma_value_cluster_pca)),'o','MarkerSize',10,'MarkerFaceColor','red');
hold on
plot(fsoma_value_cluster_pls,zeros(1,length(fsoma_value_cluster_pls)),'o','MarkerSize',10,'MarkerFaceColor','cyan');
xlabel('f_{soma} regional values','FontSize',14,'FontWeight','bold')
ylabel('counts','FontSize',14,'FontWeight','bold')
legend('','PCA cluster','PLS cluster')
grid on
saveas(fig,'dist_fsoma.svg')

fig=figure;
hist(fneurite);
hold on
plot(fneurite_value_cluster_pca,zeros(1,length(fneurite_value_cluster_pca)),'o','MarkerSize',10,'MarkerFaceColor','red');
hold on
plot(fneurite_value_cluster_pls,zeros(1,length(fneurite_value_cluster_pls)),'o','MarkerSize',10,'MarkerFaceColor','cyan');
xlabel('f_{neurite} regional values','FontSize',14,'FontWeight','bold')
ylabel('counts','FontSize',14,'FontWeight','bold')
legend('','PCA cluster','PLS cluster')
grid on
saveas(fig,'dist_fneurite.svg')

fig=figure;
hist(fextra);
hold on
plot(fextra_value_cluster_pca,zeros(1,length(fextra_value_cluster_pca)),'o','MarkerSize',10,'MarkerFaceColor','red');
hold on
plot(fextra_value_cluster_pls,zeros(1,length(fextra_value_cluster_pls)),'o','MarkerSize',10,'MarkerFaceColor','cyan');
xlabel('f_{extra} regional values','FontSize',14,'FontWeight','bold')
ylabel('counts','FontSize',14,'FontWeight','bold')
legend('','PCA cluster','PLS cluster')
grid on
saveas(fig,'dist_fextra.svg')

fig=figure;
hist(De);
hold on
plot(De_value_cluster_pca,zeros(1,length(De_value_cluster_pca)),'o','MarkerSize',10,'MarkerFaceColor','red');
hold on
plot(De_value_cluster_pls,zeros(1,length(De_value_cluster_pls)),'o','MarkerSize',10,'MarkerFaceColor','cyan');
xlabel('De regional values','FontSize',14,'FontWeight','bold')
ylabel('counts','FontSize',14,'FontWeight','bold')
legend('','PCA cluster','PLS cluster')
grid on
saveas(fig,'dist_De.svg')

fig=figure;
hist(Din);
hold on
plot(Din_value_cluster_pca,zeros(1,length(Din_value_cluster_pca)),'o','MarkerSize',10,'MarkerFaceColor','red');
hold on
plot(Din_value_cluster_pls,zeros(1,length(Din_value_cluster_pls)),'o','MarkerSize',10,'MarkerFaceColor','cyan');
xlabel('D_{in} regional values','FontSize',14,'FontWeight','bold')
ylabel('counts','FontSize',14,'FontWeight','bold')
legend('','PCA cluster','PLS cluster')
grid on
saveas(fig,'dist_Din.svg')