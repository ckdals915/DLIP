clear all; close all; clc

% input/output data (w/bias)
X = [0 0 1; 0 1 1; 1 1 1; 1 0 1];
Y = [1; 0; 1; 0];
% hyperparams setting
W0 = 2*rand(3,4)-1;
W1 = 2*rand(4,1)-1;
eta = 1;    % learning rate
iterNo = 1000000;
for i = 1:iterNo
    %forward direction
    S0 = X*W0;
    L1 = 1./(1+exp(-S0));
    S1 = L1*W1;
    Yh = 1./(1+exp(-S1));
    %error backpropagation
    dE_dS1 = -(Y-Yh).*Yh.*(1-Yh);
    dE_dL1 = dE_dS1*W1';
    dE_dS0 = dE_dL1.*L1.*(1-L1);
    % gradient descent algorithm
    dE_dW1 = L1'*dE_dS1;
    dE_dW0 = X'*dE_dS0;
    W1 = W1 - eta*dE_dW1;
    W0 = W0 - eta*dE_dW0;
end
L1 = 1./(1+exp(-X*W0));
Yh = 1./(1+exp(-L1*W1));

figure(1)
