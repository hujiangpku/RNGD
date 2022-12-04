function [data,data_test, Utruth] = generateTrainTestSplit(dataset,ratio,seed)
    normalize = false;
    %     if ~exist('seed','var')
    %         seed=42;
    %     end
    %     rng(seed)
    Utruth = [];
    if strcmp(dataset,'sarcos')
        trainDataPath = './dataset/larger_datasets/sarcos_inv.mat';
        load(trainDataPath);
    %     testDataPath = './larger_datasets/sarcos_inv_test.mat';
        numInstancePerTask = 44484;
        X = sarcos_inv(:,1:21);
        Y = sarcos_inv(:,22:28);
        numInstancePerTask = size(X,1);
        numTrain = floor(ratio*numInstancePerTask);
        N = 7;
        n = 21;
        for t=1:N
            randArr = randperm(numInstancePerTask);
            trainIndex = randArr(1:numTrain);
            testIndex = randArr((numTrain+1):end);
            if normalize
                [data(t).X,data_test(t).X] = meanVarNormalization(X(trainIndex,:),X(testIndex,:));
            else
%                 data(t).X,data_test(t).X] = meanVarNormalization(X(trainIndex,:),X(testIndex,:));
%                 [data(t).X,data_test(t).X] = meanVarNormalization(X(trainIndex,:),X(testIndex,:));
            end
            data(t).X = X(trainIndex,:);
            data(t).y = Y(trainIndex,t);
            data_test(t).X = X(testIndex,:);
            data_test(t).y = Y(testIndex,t);
        end
    elseif strcmp(dataset,'school')
        dataPath = './dataset/larger_datasets/school_b.mat';
        load(dataPath);
        N = 139;
        n = 28;
        X = x';
        Y = y;
        task_indexes(N+1) = size(Y,1)+1;
        for t=1:N
            numInstances = task_indexes(t+1)-task_indexes(t);
            numTrain = floor(ratio*numInstances);
            randArr = randperm(numInstances);
            idx = task_indexes(t):(task_indexes(t+1)-1);
            trainIndex = idx(randArr(1:numTrain));
            testIndex = idx(randArr((numTrain+1):end));
            if normalize
                [data(t).X,data_test(t).X] = meanVarNormalization(X(trainIndex,:),X(testIndex,:));
            end
            data(t).X = X(trainIndex,:);
            data(t).y = Y(trainIndex);
            data_test(t).X = X(testIndex,:);
            data_test(t).y = Y(testIndex);
        end
    elseif strcmp(dataset,'parkinson')
        dataPath = 'parkFeatures.mat';
        load(dataPath);
        N = 42;
        n = 19;
        X = x';
        Y = y1;
        task_indexes(N+1) = size(Y,1)+1;
        for t=1:N
            numInstances = task_indexes(t+1)-task_indexes(t);
            numTrain = floor(ratio*numInstances);
            randArr = randperm(numInstances);
            idx = task_indexes(t):(task_indexes(t+1)-1);
            trainIndex = idx(randArr(1:numTrain));
            testIndex = idx(randArr((numTrain+1):end));
            if normalize
                [data(t).X,data_test(t).X] = meanVarNormalization(X(trainIndex,:),X(testIndex,:));
            end
            data(t).X = X(trainIndex,:);
            data(t).y = Y(trainIndex);
            data_test(t).X = X(testIndex,:);
            data_test(t).y = Y(testIndex);
        end
    else
        N = 1000;
        n = 20;
        m = 100;
        r = 5;
        ind = randperm(N);
        ind = ind(1: N*0.8);
        
        U = randn(m,5);
        [U,~] = qr(U, 0);
        Utruth = U;
        t1 = 1; t2 = 1;
        for t = 1:N
            if min(abs(t - ind)) == 0
                data(t1).X = randn(n, m);
                w = randn(m,1);
                data(t1).y = data(t1).X*U*(U'*w) + 1e-3*norm(w)*randn(n,1);
                t1 = t1 + 1;
            else
                data_test(t2).X = randn(n, m);
                w = randn(m,1);
                data_test(t2).y = data_test(t2).X*U*(U'*w);
                t2 = t2 + 1;
            end
        end       
    end
end

function [X_train,X_test] = meanVarNormalization(X_train,X_test)
    mean_X_train = mean(X_train);
    std_X_train = std(X_train);
    idx = std_X_train==0;
    std_X_train(idx) = 1;
    X_train = bsxfun(@times,bsxfun(@minus,X_train,mean_X_train),1./std_X_train);
    X_test = bsxfun(@times,bsxfun(@minus,X_test,mean_X_train),1./std_X_train);
end

    