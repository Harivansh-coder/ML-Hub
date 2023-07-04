# docker file for building the image for the nextjs app
FROM python:3.9

# Working directory
WORKDIR /code

# Copying requirements.txt file
COPY requirements.txt /code/requirements.txt

# Installing dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copying the source code
COPY ./app /code/app

# Exposing the port
EXPOSE 8080

# Running the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]