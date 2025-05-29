# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container at /app
COPY sim_forecasting_logic.py .
COPY sim_forecasting_app.py .
# Optional: If you have a sample data file like 'Outliers.xlsx' that you want to bundle
# for easy testing or as a default example, uncomment and copy it:
# COPY Outliers.xlsx .

# Make port 8501 available to the world outside this container (Streamlit's default port)
EXPOSE 8501

# Define environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT 8501
ENV STREAMLIT_SERVER_HEADLESS true

# Run sim_forecasting_app.py when the container launches
CMD ["streamlit", "run", "sim_forecasting_app.py"]

#BUILDING A DOCKER IMAGE: "docker build -t forecasting-app ."        <---------

#RUN THE DOCKER CONTAINER: "docker run -p 8501:8501 forecasting-app" <---------