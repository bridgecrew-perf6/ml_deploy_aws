FROM python:3.6.9

LABEL maintainer='valerielim <valerieeelimyh@gmail.com>'

#set work dir
WORKDIR /ml_deploy_aws

#install dependencies
COPY ./requirements.txt /ml_deploy_aws/

# Install python requirements
RUN pip install --no-cache-dir -r requirements.txt

#copy project
COPY . . 

# expose appa port
EXPOSE 5000

ENTRYPOINT ["python"]

# run app/predictior.py when the container launches
CMD ["app/predictor.py", "run", "--host", "0.0.0.0"]