FROM python:3.6.9

LABEL maintainer='valerielim <valerieeelimyh@gmail.com>'

#set work dir
WORKDIR /ml_deploy_aws

#install dependencies
COPY ./requirements.txt /ml_deploy_aws/

# Install python requirements
RUN pip install -r requirements.txt


#copy
# COPY ./src /ml_deploy_aws/
# COPY ./app /ml_deploy_aws/
# COPY ./models/pth/resnet50 /ml_deploy_aws/models/pth/resnet50/
# COPY ./models /ml_deploy_aws/
# COPY ./tests /ml_deploy_aws/

COPY . . 

# expose appa port
EXPOSE 5000

ENTRYPOINT ["python"]

# run app/predictior.py when the container launches
CMD ["app/predictor.py", "run", "--host", "0.0.0.0"]