# Capstone Project - Azure Machine Learning Engineer (Mobile Price range Classification)

# Table of contents
1. [Project Overview ](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/main/README.md#project-overview)
2. [Overview of project pipeline](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer#overview-of-project-pipeline)
3. [Project Set Up and Installation](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer#project-set-up-and-installation)
4. [Used Dataset](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer#used-dataset)
    - [Dataset Overview](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer#dataset-overview)
    - [Project Task](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer#project-task)
    - [Dataset Access](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer#dataset-access)
5. [Automated ML](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer#automated-ml)
    - [Auto ML Experiment Results](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer#auto-ml-experiment-results)
7. [Hyperparameter Tuning in Hyperdrive](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer#hyperparameter-tuning-in-hyperdrive)
    - [Hyper drive experiment Results](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer#hyper-drive-experiment-results)   
8. [Best Model Deployment](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer#best-model-deployment)</br>
9. [Screen Recording](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer#screen-recording)</br>
10. [Standout Suggestions](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer#standout-suggestions)</br>
11. [References ](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer#references)</br>

# Project Overview 
**Problem**:
- My client want to estimate price of mobiles his company manufacture
 - **Other motivation**
    - To give tough fight to big companies like Apple,Samsung etc.
    - In this competitive mobile phone market mobile price cannot be simply assume. Data helps in more informed decisions. 
    
 **Solution** :
 - I have created and deployed mobile price range classifier model using Azure ML Studio which estimates mobile price for client who have started his own mobile company.
 - ML classifier model estimates price of mobiles his company creates based on about 20 features of mobiles
 - To solve this problem used sales data of mobile phones of various companies.
 - Try to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price.
 - I created two models: one using Automated ML (denoted as AutoML from now on) and one customized model whose hyperparameters are tuned using HyperDrive. I then compared the p    performance of both the models and deploy the best performing model
  
## Overview of project pipeline
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/main/Images/1.project_overview.png)

*TODO:* Write a short introduction to your project.
This is the final project which is the Capstone in the Udacity Azure Machine Learning Engineer Nanodegree. 
This project requires the expertise in the Azure Cloud Machine learning technologies. This acts as the final step in practically
implementing the knowledge that I have gathered from the nanodegree.

## Project Set Up and Installation
1. This project requires the creation on compute instance to run Jupyter Notebook & compute cluster to run the experiments.
2. Dataset needs to be manually selected.
3. Two experiments were run using Auto-ML & HyperDrive
4. The best model that gave good metrics was deployed and consumed.

## Used Dataset
Name: mobile_sales_data.csv

### Dataset Overview
*TODO*: Explain about the data you are using and where you got it from.
I have downloaded the dataset from [kaggle](https://www.kaggle.com/iabhishekofficial/mobile-price-classification).

  
![dataset](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/main/Images/1.reg_data.png)
### Project Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.
Task: This is a classification problem where in I'm trying to classify price of mobile in specific range. The target variable is "price_range"

Twenty (20) mobile  features:

**battery_power** : Total energy a battery can store in one time measured in mAh</br>
**blue**  : Has bluetooth or not</br>
**clock_speed**   : speed at which microprocessor executes instructions</br>
**dual_sim**  : Has dual sim support or not</br>
**fc**   : Front Camera mega pixels</br>
**four_g**  : Has 4G or not</br>
**int_memory** : Internal Memory in Gigabytes</br>
**m_dep** : Mobile Depth in cm</br>
**mobile_wt** : Weight of mobile phone</br>
**n_cores** : Number of cores of processor</br>
**pc** : Primary Camera mega pixels</br>
**px_height** : Pixel Resolution Height</br>
**px_width** : Pixel Resolution width</br>
**ram** : Random Access Memory in Megabytes</br>
**sc_h** : Screen Height of mobile in cm</br>
**sc_w** : Screen Width of mobile in cm</br>
**talk_time** : longest time that a single battery charge will last when you are</br>
**three_g**  : Has 3G or not</br>
**touch_screen**: Has touch screen or not</br>
**wifi** : Has wifi or not</br>
price_range : response variable

### Dataset Access
*TODO*: Explain how you are accessing the data in your workspace.
The dataset was downloaded from Kaggle where I have staged it for direct download to the AML workspace using SDK.

Once the dataset was downloaded, SDK was again used to clean and split the data into training and validation datasets, 
which were then stored as Pandas dataframes in memory to facilitate quick data exploration and query, and registered as AML TabularDatasets 
in the workspace to enable remote access by the AutoML experiment running on a remote compute cluster.
     ```key = 'mobile price classification dataset'
        if key in ws.datasets.keys():
        dataset = ws.datasets[key]
        print("dataset found!")
      else:
          mobile_sales_data = Dataset.get_by_name(ws, name='mobile_data')```

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
1. I created a Compute Instance with specification "STANDARD_D3_V2" to run Jupyter Notebook in Azure.
2. I have imported the dataset using TabularDataset library.
3. The configurations that I used for Auto-ML were
    ```automl_settings = {"iteration_timeout_minutes": 7,
    "experiment_timeout_minutes": 25,
    "enable_early_stopping": True,
    "primary_metric": 'accuracy',
    "featurization": 'auto',
    "verbosity": logging.INFO,
    "n_cross_validations": 4}

    # TODO: Put your automl config here
     automl_config = AutoMLConfig(compute_target = compute_target,
                              task='classification',
                              debug_log='automated_ml_errors.log',
                              training_data=mobile_sales_data,
                              label_column_name="price_range",
                              **automl_settings)```
 4. AutoML experiment settings and configuration 
   - **"iteration_timeout_minutes":** 7, time for each iteration is taken considering data size
   - **experiment_timeout_minutes: 25,**  model is able to train and optimize within 25 minute due to very small size dataset i.e 0.11 mb.
   - **"featurization": 'auto',**  By using auto, the experiment can preprocess the input data (handling missing data, converting text to numeric, etc.)
   - **n_cross_validations": 4**,  Generally it is best to take 3-5 cross validation based on various tutorials and Data camp course,it gives best results, 
                               visit [SKLEARN](https://scikit-learn.org/stable/modules/cross_validation.html)  for better understanding.
   - **primary_metric": 'accuracy'**,    considering client requirement, Accuracy is more important. Less accurate or bad classifier can make huge loss to client. 
   - **compute_target = compute_target,** compute_target is used to train model that is all processing on cloud compute i named compute as capston-compute
   - **task='classification',**          used task as classification as I have to classify price ranges for mobiles based on no of features
   - **label_column_name="price_range"**,  price_range is reponse variable which i have to estimate

### Auto ML Experiment Results
#### 1. Auto ML experiment in running state details
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/main/Images/2.auto_running.png)
#### 2. Auto ML run completed
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/main/Images/2.auto_run_complte.png)
#### 2. Best algorithms selected
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/main/Images/2.auto_algos.png)
#### 3. Best model and metrices
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/main/Images/2.auto_run_best_met.png)

## Hyperparameter Tuning in Hyperdrive
1. I have used LogisticRegression for this experiment since it is easily understandable and works well with Classification problems.
2. I have used RandomParameterSampling with 3 parameters for this model: ```RandomParameterSampling({'C': choice(0.01, 0.1, 1, 10, 100),
                                        'max_iter' : choice(50,75,100,125,150,175,200),
                                        'solver' : choice('liblinear','sag','lbfgs', 'saga')})```
3. I have used the primary metric as "Accuracy" for this problem and I have tried to maximize it.

### Hyper drive experiment Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
1. The best performing accuracy was 84%
2. The parameters of the model are: ['--C', '100', '--max_iter', '200', '--solver', 'liblinear']
3. I could increase the number of parameter ranges that I have used. I can even change the method of sampling used for the execution to run faster or slower and find good         accurate results.
#### Hyper drive experiment running Details
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/main/Images/3.hd_running.png)

#### Run completed
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/main/Images/3.hd_run.png)
#### Best model
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/main/Images/3.hd_best_run.png)

#### Hyperdrive registered model
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/main/Images/3.hd_reg_model.png)

## Best Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
As I choose Auto ML model for deployment considering Accuracy metric as main Key factor for prediction.Accuracy of Auto ML model is higher than Hyperdrive run.
#### 1. Model deployment successful 
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/main/Images/2.model_deploy.png)

#### 2. Model prediction and  sending request
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/main/Images/2.prediction.png)

#### 3.Model prediction service logs
![](https://github.com/AnshuTrivedi/Capstone-Project---Azure-Machine-Learning-Engineer/blob/main/Images/2.service%2Blogs.png)

## Screen Recording

[Video link](https://youtu.be/FjBnInTWZSc)

## Standout Suggestions
1. I explored mobile sales dataset to understand better features and granularity of data. I tried to optimize features and pre-processing such as scaling for better model           performnace.
2. Explored different tools of model training and deployment in future i will explore for image data and computer vision model deployment.

## References 
 1. [tutorila on auto train model](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-auto-train-models)
 2. [How to connect data in ML Azure](https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/how-to-connect-data-ui.md)
 3. [How to deploy model in ML Azure](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python)
 4. [How to create ML Pipelines](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-machine-learning-pipelines)
 5. [How to create register datasets](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets)
