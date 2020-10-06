# Crowdsourcing under Data Poisoning Attacks: A Comparative Study
In this project, we propose a comprehensive data poisoning attack taxonomy for truth inference in crowdsourcing and systematically evaluate the state-of-the-art truth inference methods under various data poisoning attacks. We use several evaluation metrics to evaluate and analyze the robustness or susceptibility of different methods against different attacks, which sheds light on the resilience of existing methods and ultimately helps in building more robust truth inference methods in an open setting.
Dataset: 

Paper: Crowdsourcing under Data Poisoning Attacks: A Comparative Study
https://link.springer.com/chapter/10.1007/978-3-030-49669-2_18

## Project Structure:
The main files of the project are as follows:
If you would like to run a method on a crowdsourced dataset (with workers' answers), then a direct way is to just execute run.py in the downloaded project (see Downloads section below) with four parameters: (1) 'method_file_path', (2) 'answer_file_path', (3) 'result_file_path', and (4) 'task type'. Next we introduce the four parameters in detail.

Parameters:

    'method_file_path': the path of the algorithm source file. Note that we provide all 17 algorithms' source codes (as compared in the paper) in the methods folder, and each method's source file is always the file method.py under respective folders mentioned in the table below.
    For example, if you would like to run D&S method as mentioned in the paper, then it is in fact implemented in the file methods/c_EM/method.py. Similarly for other methods, you can just replace c_EM with the corresponding folder (can be found in the table below) in the above file path.
    
    'answer_file_path': the path of the answer file. The format of answer file (i.e., a csv file) is:
(1) the first line is always 'question,worker,answer';
(2) the lines followed will be the instances of 'question,worker,answer', indicating that worker has answered question with the answer.
Next we show an example answer file: 


    'result_file_path': the path of result file, containing the inferred truth of the algorithm on the above answer file. The result file is a csv file, and each line contains a question and its inferred truth.
    'task type': it can only be Decision-Making, Single_Choice, or Numeric, indicating different task types.

Let us now give an example as follows:
python run.py methods/c_EM/method.py ./demo_answer_file.csv ./demo_result_file.csv decision-making
It will invoke D&S algorithm to read './demo_answer_file.csv' as the input (the tasks are 'decision-making' tasks), and the output will be in the file './demo_result_file.csv'.
    
    
    


* Untargeted: 
	Generating malicious workers: 
		prepareAdvData.py
			Input: 

			Output: 
			
* Targeted:





* For Targeted attack : 

* For Hueristic: 
Evaluationprogram.py (mv, ds, bcc)
C:\Users\x1\Documents\Farnaz\Research\Evaluation\mv\MVHard\data\product\Targeted_modified\result_Nov_2019\Averages
KOS, mv-hard, mv-soft
LAAS
PM 

* timization: 
Targeted: 



Program Usage:
Parameters:




 The main files of the project are as follows:

    truth inference crowdsourcing methods:
    methods folder: 	folder that stores crowdsourced truth inference methods, for experiments in Section 6.3.1.
    qualification_methods folder: 	folder that stores the truth inference methods incorporating qualification test, for experiments in Section 6.3.2.
    truth_methods folder: 	folder that stores the truth inference methods incorporating hidden test, for experiments in Section 6.3.3.
    datasets folder: folder that store all the 5 datasets used in the paper (You can refer to here, which lists our maintained crowdosurced public datasets).
    output folder: folder that stores the outputs of 3 experiments.
    exp-*.py file: python scripts for experiments.
    config.ini: configuration file for experiments.

Contacts

The project is done by ..

If you have any questions, feel free to contact ...




Paper bib:
@inproceedings{tahmasebian2020crowdsourcing,
  title={Crowdsourcing under data poisoning attacks: A comparative study},
  author={Tahmasebian, Farnaz and Xiong, Li and Sotoodeh, Mani and Sunderam, Vaidy},
  booktitle={IFIP Annual Conference on Data and Applications Security and Privacy},
  pages={310--332},
  year={2020},
  organization={Springer}
}
