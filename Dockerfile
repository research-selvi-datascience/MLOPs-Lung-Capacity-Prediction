# Uses Python 3.12 base image
FROM python:3.12

# Opens port 5000
EXPOSE 5000

# Sets container working folder
WORKDIR /opt/app2

# Copies project files into container
COPY . /opt/app2

# Installs dependencies
RUN pip install -r requirements.txt

# Runs Flask app
CMD ["python", "app/app.py"]