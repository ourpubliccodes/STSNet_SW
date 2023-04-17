* how to use folder 1:  
	To train the model on all split files in the ```./splits``` directory run this command:
	```
	python main.py --train
	```

	Results, including a copy of the split and python files, will be stored in ```./data``` directory. 
	You can specify different directory with a parameter ```-o <directory_name>``` This is convenient if you 
	are running a number of experiments and want to preserve the results and configuration. 

	The final results will be recorded in ```./data/results.txt``` with corresponding models in 
	the ```./data/models``` directory.    

	By default, the training is done with split files in ```./splits``` directory. These files were created 
	with ```create_split.py```. For example, to create 5 fold split file for the dataset run the following command:  
	```
	python create_split.py -d datasets/expression.h5 --save-dir splits --save-name exp_splits --num-splits 5
	```
	The split file will be saved as ```./splits/exp_splits.json```
	
* how to use folder 2:    
  first： set *list_file_train* and *list_file_test* in `main.py` properly, each of them is a list file, contents in file like this:  
  */home/XXX/fold/1/anger1_1/1 5 1*  
  *...*  
  where */home/XXX/fold/1/anger1_1/1* is a fold which contain a image sequence of a expression, 5 is the len of the clips, 1 is the label  
  second: set *premodel* in `main.py` if you have the pretrained model  
  third: run `python main.py` in your terminal.  
  
