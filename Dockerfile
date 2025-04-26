# Use an official Python runtime as a parent image
FROM python:3.13-slim


# Set the working directory in the container
WORKDIR /code


# Copy the requirements file into the container
COPY ./requirements/requirements.txt /code/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code into the container
COPY ./survival_model /code/app/survival_model

# Expose the port the app runs on
EXPOSE 7860

WORKDIR /code/app

# Command to run the FastAPI app
CMD ["python", "survival_model/main.py"]