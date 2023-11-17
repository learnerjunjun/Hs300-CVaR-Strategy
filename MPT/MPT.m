Bondlist=predict.Properties.VariableNames(:,1:end);
returns = readmatrix("predict.xlsx")

remean=(mean(returns)+1).^12-1
covma=(cov(returns)+1).^12-1

% 随机产生投资方案，计算并画图其可行域
rand('state',0);
%[ef_port_return,ef_port_risk,ef_port_weight]=portopt(remean,covma,50000);
weights=rand(2000,4); % 产生1000行，4列随机数
total=sum(weights,2); % 按列求和
for gpi=1:4 % 比例标准化，变成了权重矩阵
weights(:,gpi)=weights(:,gpi)./total;
end

%[portrisk,portreturn]=portstats(remean,covma,weights);
for i = 1:size(weights, 1)
    portreturn(i, :) = weights(i, :) * remean';  % 计算 weights 的第 i 行与 s 的乘积，并存储在 r 的第 i 行
    portrisk(i, :)=weights(i, :) *covma*(weights(i, :))' %计算组合方差
end

A=0.3; %风险偏好系数
%u=ef_port_risk-A*ef_port_return;
u=portreturn-A*portrisk;
[mostu,index]=max(u)
meanpr=mean(portreturn)
maxpr=max(portreturn)
meanu=mean(u)