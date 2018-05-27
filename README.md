# EmailClassification

Apurva Mithal
axm174531
CS 6375.001

			                                       R
-	Implemented spam filter using the following algorithms
	-	Naïve Bayes Classifier (log likelihood and Add-One Laplace Smoothing)
	-	Logistic Regression
	
-	System Details
	-	Operating System: Windows 10
	-	Python: 3.6

-	File Names:
	-	MultinomialNB.py – Code for Naïve Bayes Classifier
	-	LogisticRegression.py – Code for Logistic Regression

-	How to run the code
	-	Install Python 3.6
	-	Add the python.exe file to the environment variables.
 
	-	Directory Structure
		-	train
			-	spam
				-	<filename>.txt (all training spam files)
			-	ham
				-	<filename>.txt (all training ham files)
		-	test
			-	spam
				-	<filename>.txt (all training spam files)
			-	ham
				-	<filename>.txt (all training ham files)

		-	MultinomialNB.py
		-	LogisticRegression.py
		-	stop_words.txt

	-	Open the command prompt in Windows.
	-	Go to the path where the python files are present. 
	-	Run by giving the following commands:
	-	Naïve Bayes:
		-	python MultinomialNB.py
	-	Logistic Regression:
		-	Takes 2 arguments.
			-	Lambda (regularization parameter)
			-	Number of Iterations
		-	python LogisticRegression.py <lambda> <number of iterations>
			eg. python LogisticRegression.py 0.01 2

Note: Logistic regression takes time as number of iterations increases. For convenience printed the iteration number being processed. First the iteration number being processed (while learning the weight parameters) is printed for training set with stop words then for the training set without stop words. Finally, the accuracy is printed.
