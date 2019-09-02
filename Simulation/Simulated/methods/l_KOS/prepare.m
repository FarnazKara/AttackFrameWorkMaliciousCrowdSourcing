clearvars %-except filename
addpath(genpath(pwd));
fprintf("Hello")
%prepare_data(filename)

%% prepare answer Matrix
    folder = pwd;
    truthfile = fullfile(folder,"truth.csv");
    filename = fullfile(folder,"answers.csv");

    fid = fopen(filename, 'r', 'n', 'UTF-8'); % question, worker, answer
    fgetl(fid); % drop the first line
    data = textscan(fid, '%s %s %s', 'Delimiter', ',');
    fclose(fid);

    fid = fopen(truthfile, 'r', 'n', 'UTF-8'); % question, truth
    fgetl(fid); % drop the first line
    truthdata = textscan(fid, '%s %s', 'Delimiter', ',');
    fclose(fid);


    if ~exist('uniqueQ', 'var')
        uniqueQ = unique(data{1});
    end
    if ~exist('truth_labels', 'var')
        true_labels = zeros(length(uniqueQ),1);
        for i = 1:length(truthdata{1})
            Qindex = find(strcmp(uniqueQ, truthdata{1}(i)));
            true_labels(Qindex) = int8(str2double(truthdata{2}(i)));
            if true_labels(Qindex) == 1
				true_labels(Qindex) = 2;
            elseif true_labels(Qindex) == 0
				true_labels(Qindex) = 1;
            end
        end
    end
    
    uniqueW = unique(data{2});
    L = zeros(length(uniqueQ), length(uniqueW));
    for i = 1:length(data{1})
        Qindex = find(strcmp(uniqueQ, data{1}(i)));
        Windex = find(strcmp(uniqueW, data{2}(i)));
        L(Qindex, Windex) = str2double(data{3}(i)) + 1;
    end
    
    nworker = size(uniqueW);
    res_name = sprintf("kos_result_%d.csv",nworker);
    resultfile = fullfile(folder,res_name);
    clearvars -except L uniqueQ true_labels resultfile nworker;

    L(L>2) = 2;  % only for binay labels
    % true_labels(true_labels~=1) = 2;

    Model = crowd_model(L);
    verbose = 0;

    % this version closely follows the implementation in [KOS]: initializing messages to Norm(1,1), and run only for 10 iterations.
    options = {'maxIter',10, 'TOL', 0, 'initialMsg', 'ones+noise', 'verbose', verbose};
    Key_kos10 = KOS_method_crowd_model(Model, options{:});
	prob_error_kos10 = mean(Key_kos10.ans_labels ~= true_labels);  
    writecsv(resultfile, uniqueQ, Key_kos10.belTask);
    res_name = sprintf("accuracy_%d.txt",nworker);
    fileID = fopen(res_name,'w');
    fprintf(fileID,'%2.5f\n',(1-prob_error_kos10));
    fclose(fileID);
    exit
%end

