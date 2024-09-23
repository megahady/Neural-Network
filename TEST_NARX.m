% Define time series parameters
t_train = 0:0.01:4; % Time from 0 to 4 seconds for training
inputSeries_train = sin(t_train) + sin(2*t_train)+tanh(t_train); % Training data: sin(t) + sin(2t)
targetSeries_train = inputSeries_train; % Training targets (same as input for one-step prediction)

t_test = 0:0.01:4; % Time from 0 to 4 seconds for testing
inputSeries_test = sin(t_test) - 1*sin(10*t_test)./(0.1+cosh(t_test)); % Testing data: sin(t) - sin(3t)
targetSeries_test = inputSeries_test; % Testing targets

N=5;

% Define NARX network parameters
inputDelays = 1:N; % Use past 5 time steps as input
feedbackDelays = 1:N; % Use past 5 time steps of output as feedback
hiddenLayerSize = 5; % Number of neurons in hidden layer

% Create NARX network
net = narxnet(inputDelays, feedbackDelays, hiddenLayerSize);

% Prepare training data for the network
[Xs_train, Xi_train, Ai_train, Ts_train] = preparets(net, con2seq(inputSeries_train), {}, con2seq(targetSeries_train));

% Train the network on the training data
net = train(net, Xs_train, Ts_train, Xi_train, Ai_train);

% Prepare testing data for the network (without retraining)
[Xs_test, Xi_test, Ai_test, Ts_test] = preparets(net, con2seq(inputSeries_test), {}, con2seq(targetSeries_test));

% Predict the next time step for the test data
predictedSeries_test = net(Xs_test, Xi_test, Ai_test);

% Convert the predicted series from cell array to matrix for plotting
predictedSeriesMat_test = cell2mat(predictedSeries_test);

% Plot ground truth vs estimated values for the test data
figure;
plot(t_test(N+1:end), targetSeries_test(N+1:end), 'b', 'LineWidth', 1.5); % Ground truth (test data)
hold on;
plot(t_test(N+1:end), predictedSeriesMat_test, 'r--', 'LineWidth', 1.5); % Predicted series (test data)
legend('Ground Truth', 'NARX Prediction');
xlabel('Time (seconds)');
ylabel('Value');
grid on;
