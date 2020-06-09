function [bestmem, bestval, nfeval] = DE(fname, D, XVmin, XVmax, NP, itermax)
% this is the DE (differential evolution) algorithm of Rainer Storn, 
% I made some modification based on the implementation of 
% FAC Viana, SURROGATES Toolbox User's Guide, Version 3.0, 2011, available
% at http://sites.google.com/site/felipeacviana/surrogatestoolbox.
% Initial implementation can be found at http://www.icsi.berkeley.edu/~storn/code.html

% Output arguments:
% ----------------
% bestmem        parameter vector with best solution
% bestval        best objective function value
% nfeval         number of function evaluations
%
% Input arguments:
% ---------------
%
% fname          string naming a function f(x,y) to minimize
% VTR            "Value To Reach". srgtsOPTMDE will stop its minimization
%                if either the maximum number of iterations "itermax"
%                is reached or the best parameter vector "bestmem"
%                has found a value f(bestmem,y) <= VTR.
% D              number of parameters of the objective function
% XVmin          vector of lower bounds XVmin(1) ... XVmin(D)
%                of initial population
%                *** note: these are not bound constraints!! ***
% XVmax          vector of upper bounds XVmax(1) ... XVmax(D)
%                of initial population
% NP             number of population members
% itermax        maximum number of iterations (generations)
% F              DE-stepsize F from interval [0, 2]
% CR             crossover probability constant from interval [0, 1]
% strategy       1 --> DE/best/1/exp           6 --> DE/best/1/bin
%                2 --> DE/rand/1/exp           7 --> DE/rand/1/bin
%                3 --> DE/rand-to-best/1/exp   8 --> DE/rand-to-best/1/bin
%                4 --> DE/best/2/exp           9 --> DE/best/2/bin
%                5 --> DE/rand/2/exp           else  DE/rand/2/bin
%                Experiments suggest that /bin likes to have a slightly
%                larger CR than /exp.
% refresh        intermediate output will be produced after "refresh"
%                iterations. No intermediate output will be produced
%                if refresh is < 1
%
%       The first four arguments are essential (though they have
%       default values, too). In particular, the algorithm seems to
%       work well only if [XVmin,XVmax] covers the region where the
%       global minimum is expected. DE is also somewhat sensitive to
%       the choice of the stepsize F. A good initial guess is to
%       choose F from interval [0.5, 1], e.g. 0.8. CR, the crossover
%       probability constant from interval [0, 1] helps to maintain
%       the diversity of the population and is rather uncritical. The
%       number of population members NP is also not very critical. A
%       good initial guess is 10*D. Depending on the difficulty of the
%       problem NP can be lower than 10*D or must be higher than 10*D
%       to achieve convergence.
%       If the parameters are correlated, high values of CR work better.
%       The reverse is true for no correlation.
%
% Differential Evolution for MATLAB
% Copyright (C) 1996, 1997 R. Storn
% International Computer Science Institute (ICSI)
% 1947 Center Street, Suite 600
% Berkeley, CA 94704
% E-mail: storn@icsi.berkeley.edu
% WWW:    http://http.icsi.berkeley.edu/~storn
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 1, or (at your option)
% any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details. A copy of the GNU
% General Public License can be obtained from the
% Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

%--------------------------------------------------
% Check input variables
VTR = -Inf;
refresh =0;
F = 0.8;
CR = 0.8;
strategy = 6;
use_vectorize = 1;

if NP < 5
    NP = 5;
    fprintf(1,' NP increased to minimal value 5\n');
end
if ( (CR < 0) || (CR > 1) )
    CR=0.5;
    fprintf(1,'CR should be from interval [0,1]; set to default value 0.5\n');
end
if itermax <= 0
    itermax = 200;
    fprintf(1,'itermax should be > 0; set to default value 200\n');
end
%-------------------------------------------------------------------------
% Initialize population and some arrays
% pop is a matrix of size NPxD. It will be initialized with random
% values between the min and max values of the parameters
pop= repmat(XVmin,NP,1) + rand(NP,D).*repmat(XVmax - XVmin,NP,1);
% create and reset the "cost array"
val       = zeros(1,NP);
% number of function evaluations
nfeval    = 0;
%-------------------------------------------------------------------------
% Evaluate the best member after initialization
if use_vectorize == 1
    val = feval(fname, pop);
    nfeval = nfeval + size(pop, 1);
    [bestval, ibest] = min(val);
else
    ibest   = 1;                      % start with first population member
    val(1)  = feval(fname,pop(ibest,:));
    bestval = val(1);                 % best objective function value so far
    nfeval  = nfeval + 1;
    for i=2:NP                        % check the remaining members
        val(i) = feval(fname,pop(i,:));
        nfeval  = nfeval + 1;
        if (val(i) < bestval)           % if member is better
            ibest   = i;                 % save its location
            bestval = val(i);
        end
    end
end



% best member of current iteration
bestmemit = pop(ibest,:);
% best member ever
bestmem   = bestmemit;
%-------------------------------------------------------------------------
% DE-Minimization
% popold is the population which has to compete. It is static through one
% iteration. pop is the newly emerging population
% initialize bestmember  matrix
bm  = zeros(NP,D);
% intermediate population of perturbed vectors
ui  = zeros(NP,D);
% rotating index array (size NP)
rot = (0:1:NP-1);
% rotating index array (size D)
rotd= (0:1:D-1);
XVmin = repmat(XVmin, NP,1);
XVmax = repmat(XVmax, NP,1);
iter = 1;
% the while loop
while ((iter < itermax) && (bestval > VTR))
    % save the old population
    popold = pop;
    % index pointer array
    ind = randperm(4);
    % shuffle locations of vectors
    a1  = randperm(NP);
    % rotate indices by ind(1) positions
    rt = rem(rot+ind(1),NP);
    % rotate vector locations
    a2  = a1(rt+1);
    rt = rem(rot+ind(2),NP);
    a3  = a2(rt+1);
    rt = rem(rot+ind(3),NP);
    a4  = a3(rt+1);
    rt = rem(rot+ind(4),NP);
    a5  = a4(rt+1);
    % shuffled population 1
    pm1 = popold(a1,:);
    % shuffled population 2
    pm2 = popold(a2,:);
    % shuffled population 3
    pm3 = popold(a3,:);
    % shuffled population 4
    pm4 = popold(a4,:);
    % shuffled population 5
    pm5 = popold(a5,:);
    % population filled with the best member of the last iteration
    for i=1:NP
        bm(i,:) = bestmemit;
    end
    % all random numbers < CR are 1, 0 otherwise
    mui = rand(NP,D) < CR;
    % binomial crossover
    if (strategy > 5)
        st = strategy-5;
    else
        % exponential crossover
        st = strategy;
        % transpose, collect 1's in each column
        mui=sort(mui,2)';
        for i=1:NP
            n=floor(rand*D);
            if n > 0
                rtd = rem(rotd+n,D);
                %rotate column i by n
                mui(:,i) = mui(rtd+1,i);
            end
        end
        % transpose back
        mui = mui';
    end
    % inverse mask to mui
    mpo = mui < 0.5;
    
    % strategy       1 --> DE/best/1/exp                     6 --> DE/best/1/bin
    %                        2 --> DE/rand/1/exp                    7 --> DE/rand/1/bin
    %                        3 --> DE/rand-to-best/1/exp     8 --> DE/rand-to-best/1/bin
    %                        4 --> DE/best/2/exp                     9 --> DE/best/2/bin
    %                        5 --> DE/rand/2/exp                     else  DE/rand/2/bin
    
    switch st
        % DE/best/1
        case 1
            % differential variation
            ui = bm + F*(pm1 - pm2);
            % crossover
            ui = popold.*mpo + ui.*mui;
            % DE/rand/1
        case 2
            % differential variation
            ui = pm3 + F*(pm1 - pm2);
            % crossover
            ui = popold.*mpo + ui.*mui;
            % DE/rand-to-best/1
        case 3
            ui = popold + F*(bm-popold) + F*(pm1 - pm2);
            % crossover
            ui = popold.*mpo + ui.*mui;
            % DE/best/2
        case 4
            % differential variation
            ui = bm + F*(pm1 - pm2 + pm3 - pm4);
            % crossover
            ui = popold.*mpo + ui.*mui;
            % DE/rand/2
        case 5
            % differential variation
            ui = pm5 + F*(pm1 - pm2 + pm3 - pm4);
            % crossover
            ui = popold.*mpo + ui.*mui;
    end
    
    % correcting violations on the lower bounds of the variables
    % these are good to go
    maskLB = ui > XVmin;
    % these are good to go
    maskUB = ui < XVmax;
    ui     = ui.*maskLB.*maskUB + XVmin.*(~maskLB) + XVmax.*(~maskUB);
    
    %%-------------------------------------------------------------------------
    % Select which vectors are allowed to enter the new population
    if use_vectorize ==1
        tempval = feval(fname, ui);
        nfeval = nfeval + size(ui,1);
        % if competitor is better than value in "cost array"
        indx = tempval <= val;
        % replace old vector with new one (for new iteration)
        pop(indx, :) = ui(indx, :);
        val(indx, :) = tempval(indx, :);
        %we update bestval only in case of success to save time
        indx = tempval < bestval;
        if sum(indx)~=0
            [bestval, ind] = min(tempval);
            bestmem = ui(ind,:);
        end
    else
        for i=1:NP
            % check cost of competitor
            tempval = feval(fname,ui(i,:));
            nfeval  = nfeval + 1;
            % if competitor is better than value in "cost array"
            if (tempval <= val(i))
                % replace old vector with new one (for new iteration)
                pop(i,:) = ui(i,:);
                % save value in "cost array"
                val(i)   = tempval;
                % we update bestval only in case of success to save time
                % if competitor better than the best one ever
                if (tempval < bestval)
                    % new best value
                    bestval = tempval;
                    % new best parameter vector ever
                    bestmem = ui(i,:);
                end
            end
        end
        
    end
    
    
    % freeze the best member of this iteration for the coming
     % iteration. This is needed for some of the strategies.
    bestmemit = bestmem;
    % print the information to screen
    if refresh == 1
        fprintf('Iteration: %d,  Best: %f,  F: %f,  CR: %f,  NP: %d\n',iter,bestval,F,CR,NP);
    end    
    iter = iter + 1;
end %---end while ((iter < itermax) ...
