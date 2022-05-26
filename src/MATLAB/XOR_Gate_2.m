X = [0 0; 0 1; 1 1; 1 0];
b = [1; 1; 1; 1];
X = [X b];
% Y = [0;1;0;1];
Y = [1 0; 0 1; 1 0; 0 1];

% model
W0 = 2*rand(3,4) - 1;
W1 = 2*rand(4,2) - 1;   % Classification of 4X2 Y

iterNo = 10000;
eta = 1;
for i = 1:iterNo
    % forward
    S0 = X * W0;
    L1 = 1./(1+exp(-S0));
    S1 = L1*W1;
    Yh = 1./(1+exp(-S1));

    % back Propegation
    dE_dYh = -(Y-Yh);
    dE_dS1 = dE_dYh .* Yh .* (1-Yh);
    dE_dL1 = dE_dS1 * W1';
    dE_dS0 = dE_dL1 .* L1 .* (1-L1);

    % Update
    dE_dW1 = L1' * dE_dS1;
    dE_dW0 = X' * dE_dS0;
    W1 = W1 - eta * dE_dW1;
    W0 = W0 - eta * dE_dW0;
end

S0 = X * W0;
    L1 = 1./(1+exp(-S0));
    S1 = L1*W1;
    Yh = 1./(1+exp(-S1))