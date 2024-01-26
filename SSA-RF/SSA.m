function [Best_pos, Best_score, curve, avcurve] = SSA(pop, Max_iter, lb, ub, dim, fobj)
disp('SSA starts working')
%%  parameter settings      
ST = 0.8;                    
PD = 0.2;                   
PDNumber = pop * PD;         
SDNumber = pop - pop * PD;   

%% Determine the number of optimization parameters
if(max(size(ub)) == 1)
   ub = ub .* ones(1, dim);
   lb = lb .* ones(1, dim);  
end

%%  Population initialization
pop_lsat = initialization(pop, dim, ub, lb);
pop_new  = pop_lsat;

%%  Calculate initial fitness value
fitness = zeros(1, pop);
for i = 1 : pop
   fitness(i) =  fobj(pop_new(i, :));
end

%%  Get the global optimal fitness value
[fitness, index]= sort(fitness);
GBestF = fitness(1); 

%%  Get the global optimal population
for i = 1 : pop
    pop_new(i, :) = pop_lsat(index(i), :);
end

GBestX = pop_new(1, :);
X_new  = pop_new;

%% optimization
for i = 1: Max_iter

   BestF = fitness(1);
   R2 = rand(1);

   for j = 1 : PDNumber
      if(R2 < ST)
          X_new(j, :) = pop_new(j, :) .* exp(-j / (rand(1) * Max_iter));
      else
          X_new(j, :) = pop_new(j, :) + randn() * ones(1, dim);
      end     
   end
   
   for j = PDNumber + 1 : pop
        if(j > (pop - PDNumber) / 2 + PDNumber)
          X_new(j, :) = randn() .* exp((pop_new(end, :) - pop_new(j, :)) / j^2);
        else
          A = ones(1, dim);
          for a = 1 : dim
              if(rand() > 0.5)
                A(a) = -1;
              end
          end
          AA = A' / (A * A');     
          X_new(j, :) = pop_new(1, :) + abs(pop_new(j, :) - pop_new(1, :)) .* AA';
       end
   end
   
   Temp = randperm(pop);
   SDchooseIndex = Temp(1 : SDNumber); 
   
   for j = 1 : SDNumber
       if(fitness(SDchooseIndex(j)) > BestF)
           X_new(SDchooseIndex(j), :) = pop_new(1, :) + randn() .* abs(pop_new(SDchooseIndex(j), :) - pop_new(1, :));
       elseif(fitness(SDchooseIndex(j)) == BestF)
           K = 2 * rand() -1;
           X_new(SDchooseIndex(j), :) = pop_new(SDchooseIndex(j), :) + K .* (abs(pop_new(SDchooseIndex(j), :) - ...
               pop_new(end, :)) ./ (fitness(SDchooseIndex(j)) - fitness(end) + 10^-8));
       end
   end

%%  border control
   for j = 1 : pop
       for a = 1 : dim
           if(X_new(j, a) > ub(a))
              X_new(j, a) = ub(a);
           end
           if(X_new(j, a) < lb(a))
              X_new(j, a) = lb(a);
           end
       end
   end 

%%  Get fitness value
   for j = 1 : pop
    fitness_new(j) = fobj(X_new(j, :));
    
   end
   
%%  Get the optimal population
   for j = 1 : pop
       if(fitness_new(j) < GBestF)
          GBestF = fitness_new(j);
          GBestX = X_new(j, :);
          
       end
   end
  
%%  Update population and fitness values
   pop_new = X_new;
   fitness = fitness_new;
  
   disp(['At iteration',   num2str(i)  ,'the best fitness is',  num2str(GBestF)]);
%%  update population
   [fitness, index] = sort(fitness);
   for j = 1 : pop
      pop_new(j, :) = pop_new(index(j), :);
   end

%%  Get optimization curve
   curve(i) = GBestF;
   avcurve(i) = sum(curve) / length(curve);
end

%%  get optimal value
Best_pos = [round(GBestX(1: 2))] ;
Best_score = curve(end);

