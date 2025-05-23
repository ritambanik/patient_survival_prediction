# Use an official Python runtime as a parent image
FROM python:3.13-slim


# Set the working directory in the container
WORKDIR /code


# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install fastapi[standard] uvicorn


# Copy the rest of the application code into the container
COPY ./app /code/app

# Expose the port the app runs on
EXPOSE 8000

WORKDIR /code/app

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]