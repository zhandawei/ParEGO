% 1. The ParEGO algorithm decomposes a multiobjective problem into multiple
%     single-objective problems and solves one sinlge-objective problem
%     randomly in each iteration. 
% 2. The dace toolbox [2] is used for building the Kriging models in the
%    implementations.
% 3. The non-dominated sorting method by Yi Cao [3] is used to identify the
%    non-dominated fronts from all the design points
% 5. The EIM criteria are maximized by DE [5] algorithm.
% [1]  Knowles, J. ParEGO: A hybrid algorithm with on-line landscape 
%      approximation for expensive multiobjective optimization problems.
%     IEEE Transactions on Evolutionary Computation, 2006,10(1):50-66.
% [2] Lophaven SN, Nielsen HB, and Sodergaard J, DACE - A MATLAB Kriging
%     Toolbox, Technical Report IMM-TR-2002-12, Informatics and Mathematical
%     Modelling, Technical University of Denmark, 2002. Available at:
%     http://www2.imm.dtu.dk/~hbn/dace/.
% [3] http://www.mathworks.com/matlabcentral/fileexchange/17251-
%      pareto-front.
% [4] N. Beume, C.M. Fonseca, M. Lopez-Ibanez, L. Paquete, J. Vahrenhold,
%     On the Complexity of Computing the Hypervolume Indicator, IEEE
%     Transactions on Evolutionary Computation,2009,13(5):1075-1082.
% zhandawei@swjtu{dot}edu{dot}cn
% 2019.09.18 initial creation
% -----------------------------------------------------------------------------------------
clearvars;close all;
% settings of the problem
% for ZDT test problems, the number of objectives should be 2
fun_name = 'ZDT1';
% number of objectives
num_obj = 2;
% number of design variables
num_vari = 10;
% number of initial design points
num_initial = 100;
% the maximum allowed evaluations
max_evaluation = 200;
% get the information about the problem
switch fun_name
    case {'ZDT1', 'ZDT2', 'ZDT3'}
        design_space=[zeros(1,num_vari);ones(1,num_vari)]; ref_point = 11*ones(1,2);
    case {'DTLZ2','DTLZ5'}
        design_space=[zeros(1,num_vari);ones(1,num_vari)]; ref_point = 2.5*ones(1,num_obj);
    case 'DTLZ7'
        design_space=[zeros(1,num_vari);ones(1,num_vari)]; ref_point = (num_obj+1)*10*ones(1,num_obj);
    otherwise
        error('objective function is not defined!')
end
% generate weight vectors
if num_obj == 2
    num_weight = 11;
elseif num_obj == 3
    num_weight = 15;
else
    num_weight = 56;
end
weight = UniformPoint(num_weight,num_obj);
% the intial design points, points sampled all at once
sample_x = repmat(design_space(1,:),num_initial,1) + repmat(design_space(2,:)-design_space(1,:),num_initial,1).*lhsdesign(num_initial,num_vari,'criterion','maximin','iterations',1000);
sample_y = feval(fun_name, sample_x, num_obj);
hypervolume = zeros(max_evaluation-num_initial+1,1);
iteration = 0;
evaluation = size(sample_x,1);
% calculate the initial hypervolume values and print them on the screen
index = Paretoset(sample_y);
non_dominated_front = sample_y(index,:);
hypervolume(1) = Hypervolume(non_dominated_front,ref_point);
% plot current non-dominated front points
if num_obj == 2
    scatter(non_dominated_front(:,1), non_dominated_front(:,2),'ro', 'filled');title(sprintf('iteration: %d, evaluations: %d',0,evaluation));drawnow;
elseif num_obj == 3
    scatter3(non_dominated_front(:,1), non_dominated_front(:,2),non_dominated_front(:,3),'ro', 'filled');
    title(sprintf('iteration: %d, evaluations: %d',0,evaluation));drawnow;
end
% print the hypervolume information
fprintf('ParEGO on %s problem, iteration: %d, evaluation: %d, hypervolume: %f \n', fun_name,iteration, evaluation, hypervolume(1));
%-------------------------------------------------------------------------
% beginning of the iteration
while evaluation < max_evaluation
    % randomly select a weight vector
    lamda  = weight(randi(size(weight,1)),:);
    % build the weighted objective function
    sample_y_scaled = (sample_y - min(sample_y))./(max(sample_y) - min(sample_y));
    sample_y_pcheby = max(sample_y_scaled.*lamda,[],2) + 0.05*sum(sample_y_scaled.*lamda,2);
    % build (re-build) the initial Kriging models
    kriging_obj = dacefit(sample_x,sample_y_pcheby,'regpoly0','corrgauss',1*ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
    infill_criterion = @(x)Infill_EI(x, kriging_obj, min(sample_y_pcheby));
    % a genetic algorithm is used for the maximization problem
    [best_x,best_EI] = Optimizer_GA(infill_criterion, num_vari, design_space(1,:), design_space(2,:), 100, 100);
    % do the expensive evaluations
    best_y = feval(fun_name, best_x, num_obj);
    evaluation = evaluation + size(best_y,1);
    iteration = iteration + 1;
    % add the evaluated points to design set
    sample_x = [sample_x;best_x];
    sample_y = [sample_y;best_y];
    % plot current non-dominated front points
    index = Paretoset(sample_y);
    non_dominated_front = sample_y(index,:);
    hypervolume(iteration) = Hypervolume(non_dominated_front,ref_point);
    fprintf('ParEGO on %s problem, iteration: %d, evaluation: %d, hypervolume: %f \n', fun_name,iteration, evaluation, hypervolume(iteration));
    % plot current non-dominated front points
    if num_obj == 2
        scatter(non_dominated_front(:,1), non_dominated_front(:,2),'ro', 'filled');title(sprintf('iteration: %d, evaluations: %d',iteration,evaluation));drawnow;
    elseif num_obj == 3
        scatter3(non_dominated_front(:,1), non_dominated_front(:,2),non_dominated_front(:,3),'ro', 'filled');
        title(sprintf('iteration: %d, evaluations: %d',iteration,evaluation));drawnow;
    end
end
