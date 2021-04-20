# Capstone Project - Azure Machine Learning Engineer 
Mobile Price range Classification

*TODO:* Write a short introduction to your project.
This is the final project which is the Capstone in the Udacity Azure Machine Learning Engineer Nanodegree. 
This project requires the expertise in the Azure Cloud Machine learning technologies. This acts as the final step in practically
implementing the knowledge that I have gathered from the nanodegree. I created and deployed mobile price range classifier model.

## Project Set Up and Installation
1. This project requires the creation on compute instance to run Jupyter Notebook & compute cluster to run the experiments.
2. Dataset needs to be manually selected.
3. Two experiments were run using Auto-ML & HyperDrive
4. The best model that gave good metrics was deployed and consumed.

## Dataset
Name: mobile_sales_data

### Overview
*TODO*: Explain about the data you are using and where you got it from.
I have downloaded the dataset from [kaggle](https://www.kaggle.com/iabhishekofficial/mobile-price-classification).

  **context:**
  Bob has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc.
  He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he         collects sales data of mobile phones of various companies.
  Bob wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So I       helped him to solve this problem by creating ML classifier model.
![dataset](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/master/Images/1.registered_data.png)
### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.
Task: This is a classification problem where in I'm trying to classify price of mobile in specific range. The target variable is "price_range"

Twenty (20) mobile  features:

battery_power : Total energy a battery can store in one time measured in mAh</br>
blue  : Has bluetooth or not</br>
clock_speed   : speed at which microprocessor executes instructions</br>
dual_sim  : Has dual sim support or not</br>
fc   : Front Camera mega pixels</br>
four_g  : Has 4G or not</br>
int_memory : Internal Memory in Gigabytes</br>
m_dep : Mobile Depth in cm</br>
mobile_wt : Weight of mobile phone</br>
n_cores : Number of cores of processor</br>
pc : Primary Camera mega pixels</br>
px_height : Pixel Resolution Height</br>
px_width : Pixel Resolution width</br>
ram : Random Access Memory in Megabytes</br>
sc_h : Screen Height of mobile in cm</br>
sc_w : Screen Width of mobile in cm</br>
talk_time : longest time that a single battery charge will last when you are</br>
three_g  : Has 3G or not</br>
touch_screen: Has touch screen or not</br>
wifi : Has wifi or not</br>
price_range : 

### Access
*TODO*: Explain how you are accessing the data in your workspace.
The dataset was downloaded from Kaggle where I have staged it for direct download to the AML workspace using SDK.

Once the dataset was downloaded, SDK was again used to clean and split the data into training and validation datasets, 
which were then stored as Pandas dataframes in memory to facilitate quick data exploration and query, and registered as AML TabularDatasets 
in the workspace to enable remote access by the AutoML experiment running on a remote compute cluster.
![]()

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
1. I created a Compute Instance with specification "STANDARD_D3_V2" to run Jupyter Notebook in Azure.
2. I have imported the dataset using TabularDataset library.
3. The configurations that I used for Auto-ML were
    ```automl_config = AutoMLConfig(compute_target = compute_target,
                             task='classification',
                             debug_log='automated_ml_errors.log',
                             training_data=train_data,
                             label_column_name="price_range",
                             **automl_settings)```

### Results
#### 1. Run Details
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/master/Images/2.automl_run_complete.png)
#### 2. Best algorithms selected
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/master/Images/2.best_algos.png)
#### 3. Best model
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/master/Images/2.auto_best_model.png)
## Hyperparameter Tuning
1. I have used LogisticRegression for this experiment since it is easily understandable and works well with Classification problems.
2. I have used RandomParameterSampling with 3 parameters for this model: ```RandomParameterSampling({'C': choice(0.01, 0.1, 1, 10, 100),
                                        'max_iter' : choice(50,75,100,125,150,175,200),
                                        'solver' : choice('liblinear','sag','lbfgs', 'saga')})```
3. I have used the primary metric as "Accuracy" for this problem and I have tried to maximize it.

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
1. The best performing accuracy was 90%
2. The parameters of the model are: ['--C', '0.1', '--max_iter', '50', '--solver', 'liblinear']
3. I could increase the number of parameter ranges that I have used. I can even change the method of sampling used for the execution to run faster or slower and find good         accurate results.
####Run Details
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/master/Images/3.hd_run_completed.png)
#### Best model
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/master/Images/3.hd_best_run.png)
## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/master/Images/2.model_deployed.png)
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/master/Images/2.service_logs.png)

## Screen Recording

[Video link](https://youtu.be/xZ4158NGQBs)

## References 
 1. [tutorila on auto train model](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-auto-train-models)
 2. [How to connect data in ML Azure](https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/how-to-connect-data-ui.md)
 3. [How to deploy model in ML Azure](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python)
 4. [How to create ML Pipelines](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-machine-learning-pipelines)
 5. [How to create register datasets](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets)
