# Dockerize and deploy machine learning application on AWS
- A multi-class image classification use case
- Tools: Pytorch, ResNet, Flask, Unit test, EC2, Jenkins, Github
- work in progress

Step by step guide 

## Part 1: Develop model using transfer learning
0. create a virtual environment, install the req and activate the venv

```
(assuming you're also using anaconda3's python) python -m venv <venvName>
pip install requirements.txt
source <venvName>/bin/activate
```

Images were downloaded from https://www.kaggle.com/clorichel/boat-types-recognition 

1. Create data folder strucutre 

```
python src/pytorch_data_prep.py  --jobname vl_test1 --original_data_dir 'data/raw/' --test_ratio 0.200000

python src/pytorch_data_prep.py  --jobname vl_testSmall1 --original_data_dir 'data/rawSmall/' --test_ratio 0.200000
```

2. Train

```
python src/train.py  --jobname vl_test_1 --new_data_dir 'data/processed' --result_base_dir 'src/output'  --epochs 50 --max_epochs_stop 10 --img_size 256 --cnn_sel 'resnet50' --batch_size 32

```


3. Inference

```
#small
python src/predict.py --jobname vl_blindTest --model_fn src/output/vl_test_1/resnet50-transfer.pth --test_dir data/processed/test --result_base_dir 'src/output' --img_size 256 

python src/predict.py --jobname vl_blindTestSmall --model_fn src/output/vl_testSmall_1/resnet50-transfer.pth --test_dir data/processedSmall/test --result_base_dir 'src/output' --img_size 256 

python src/predict.py --jobname vl_blindTest --model_fn src/output/vl_test_1/resnet50-transfer.pth --test_dir ./app/static/uploads/b_20201211_100510 --result_base_dir 'src/output' --img_size 256 
```

4. flask

```
python app/predictor.py
```

5. Build the docker image (make sure you have docker running). 
It’s not mandatory to specify a tag name. The :latest tag is a default tag when build is run without a specific tag specified. explicitly tag your image after each build if you want to maintain a good version history.

```
docker build -t valerielimyh/ml_deploy_aws:1.0 .
```
- the “.” at the end of the command tells Docker to locate the Dockerfile in my current directory, which is my project folder. 

a. Use command ‘docker images’ to see a docker image with a docker repository named `ml_deploy_aws` created. (Another repository named python will also be seen since it is the base image on top of which we build our custom image.)

6. Now, the image is built and ready to be run locally

```
docker run --name cv-deploy -p 5000:5000 valerielimyh/ml_deploy_aws:1.0
```

(Optional) Clean up the container

```
docker rmi ml_deploy_aws
```

7. Push docker image to Docker Hub
a. If you’re doing this for the first time, you’ll have to log in to Docker Hub using the command below

```
docker login --username=<yourhubusername>
```

b. copy the IMAGE ID for that particular image and tag it

```
docker tag 689e0e8ba525 valerielimyh/ml_deploy_aws:1.0
```

c. push your image to Docker Hub using the repository you created with the command

```
docker push valerielimyh/ml_deploy_aws
```

## Part 2: Run the Docker Image on the EC2 instance

7. ssh into your EC2 instance (I'm using  Amazon Linux AMI 2018.03.0 (HVM))
 
```
ssh -i ~/.ssh/yourpairkey.pem ec2-user@my-instance-public-dns-name
```

8. in your EC2 instance,  update your instance packages

```
sudo yum update
sudo yum install docker
```

9. After installation, pull the docker image we pushed to the repository.

```
docker pull valerielimyh/ml_deploy_aws:1.0
```

a. if you face this error `Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?`, resolve by doing

```
sudo service docker start
Then pull the image again.
```

10. Confirm image is downloaded by running

```
docker images
```

11. Run the docker image 

```
docker run --name cv-deploy -d -p 5000:5000 valerielimyh/ml_deploy_aws:1.0
```

You can test the API on your browser with the Public DNS for your instance 


## Part 3: Install and Configure Jenkins
12. install java1.8 

```
sudo yum install java-1.8.0
```

13.  ensure that your software packages are up to date on your instance

```
sudo yum update –y
```

14. Add the Jenkins repo 

```
sudo wget -O /etc/yum.repos.d/jenkins.repo http://pkg.jenkins-ci.org/redhat/jenkins.repo
```

15.  Import a key file from Jenkins-CI to enable installation from the package

```
sudo rpm --import https://jenkins-ci.org/redhat/jenkins-ci.org.key
```

16. Install Jenkins

```
sudo yum install jenkins -y
```

17. Change settings to JENKINS_USER="root"

```
sudo vim /etc/sysconfig/jenkins
(`i` to insert; `esc` + `:wq` to write and quit)
```

18. Start Jenkins as a service.

```
sudo service jenkins start
```

19. To start the jenkins service at boot-up

```
sudo systemctl enable jenkins.service
```

20. 

```
sudo systemctl enable jenkins.service
```

21. On browser, open the following link
http://<yourPublicDNS>:8080
- can be your server IP or public DNS at the above port to get the jenkins dashboard by making sure that the port is open i.e. you added the inbould rule for that port on your AMI server (default port 8080)

21. on server, copy the initial password to proceed with the installation

```
sudo cat /var/lib/jenkins/secrets/initialAdminPassword
```

22. On browser, 
a. install suggested plugins. 
b. Sign up your account. 
c. Save and go ahead. 
d. Go to ‘Jenkins management’

23. On EC2, install git; and check where it's installed
a. sudo yum install git
b. which git

24. Goto to -> Manage Jenkins -> Global Tool Configuration ->Git->Path to Git executable and copy the git executable path. Mine is /usr/bin/git

25. Follow [this guide](https://towardsdatascience.com/automating-data-science-projects-with-jenkins-8e843771aa02) to build Jenkins pipeline 



Reference 
https://medium.com/@mohan08p/install-and-configure-jenkins-on-amazon-ami-8617f0816444 

Security settings
http://abhijitkakade.com/2019/06/how-to-reset-jenkins-admin-users-password/