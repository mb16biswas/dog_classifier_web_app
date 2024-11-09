
# dog classifier web app

* Build an end to end Deep learning application, which can classify 120 different dog breeds.

* Used models from tensoflow applications and tensorflow hub , later used the best model in my case mobilenet_v2_130_224 and later added costume layers for our prediction 

* the front end of is created with Streamlit 
 


# Deployed Here
[Live Link](https://dog-cls-deploy-1005503367247.asia-south1.run.app/)


# Download the repository and install the required packages:

Clone this repo and install the required packages:

`pip3 install -r requirements.txt`

To run streamlit app

`streamlit run app.py`




Docker commands for Development:

`docker build -f Dockerfile.dev -t app:latest .`

`docker run -p 8000:8000 app:latest`

# For GCP Production:

`gcloud auth login`

`gcloud config set project PROJECT_ID`

`gcloud builds submit --tag gcr.io/<ProjectName>/<AppName>  --project=<ProjectName>`

`gcloud run deploy --image gcr.io/<ProjectName>/<AppName> --platform managed  --project=<ProjectName> --allow-unauthenticated`


<!---https://user-images.githubusercontent.com/64213233/137639122-529cf04c-d82a-47f3-aa31-ca1fdc3a46df.mp4--->

# Demo

https://user-images.githubusercontent.com/64213233/217749217-7e3c7c1f-a015-402b-82f2-e2351267a5da.mp4


