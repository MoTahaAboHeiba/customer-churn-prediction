# Customer Churn Prediction

## Project Overview
This project focuses on developing a predictive model to identify customers who are likely to discontinue their relationship with the company (churn). By accurately forecasting customer churn, the company can implement proactive retention strategies, thereby enhancing customer satisfaction and safeguarding revenue streams.

## Key Features
- Comprehensive data loading and exploratory data analysis to gain insights into the dataset  
- Rigorous data cleaning and preprocessing to ensure high-quality inputs  
- Implementation and evaluation of various classification algorithms, including:
  - Logistic Regression  
  - Support Vector Machine  
  - Random Forest  
  - K-Nearest Neighbors  
- Model comparison to select the most effective classifier for the task  
- An interactive user interface built using Streamlit, allowing users to easily engage with the prediction model  
- Containerization of the application utilizing Docker and Docker Compose for streamlined deployment and scalability  
- Docker image hosted on DockerHub for convenient access and distribution  

## Technology Stack
- Python 3.9  
- Streamlit (User Interface)  
- Scikit-learn (Machine Learning)  
- Docker and Docker Compose (Containerization and orchestration)  

## Setup Instructions

1. Clone the repository:  
  -  git clone https://github.com/MoTahaAboHeiba/customer-churn-prediction.git
     cd customer-churn-prediction
2. Build the Docker image :
  - docker build -t your_dockerhub_username/churn-prediction.
3. Launch the application using Docker Compose:
  - docker-compose up
4. Access the application by navigating to:
  - http://localhost:8501


## Project Team
* Mohamed Taha Hassan

* Hamdi Saad Hamed

* Mohamed Farag

* Mohamed Ali AbdulNabi Ali

* Mohamed Mohsen Awadullah
  
