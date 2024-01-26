function Positions = initialization(SearchAgents_no, dim, ub, lb)

%%  初始化

%%  待优化参数个数
Boundary_no = size(ub, 2); 

%%  若待优化参数个数为1
if Boundary_no == 1
    Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
end

%%  如果存在多个输入边界个数
if Boundary_no > 1
    for i = 1 : dim
        ub_i = ub(i);
        lb_i = lb(i);
        Positions(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
    end
end