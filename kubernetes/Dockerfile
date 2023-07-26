# set env variables
ARG version=2021.7.1-debian-11-r6
ARG DEBIAN_FRONTEND=noninteractive

# pull binami scikit intel image
FROM bitnami/scikit-learn-intel:$version

# copy assets over to image
COPY /app /app

# set the working directory
WORKDIR /app
ENV PATH=/.local/bin:$PATH
RUN pip3 install --user --no-cache-dir -r /app/requirements.txt

# export port for Load Balancer
EXPOSE 5000

ENTRYPOINT ["python", "server.py"]