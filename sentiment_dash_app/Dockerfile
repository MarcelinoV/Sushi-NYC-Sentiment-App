# Step 1: Use a lightweight Python base image
FROM python:3.12-slim

# Step 2: Set the working directory
WORKDIR /app

# Step 3: Copy requirements file and install dependencies
COPY requirements.txt /app/

# Update the package manager and install necessary system packages
RUN pip install -r requirements.txt

# Step 4: Copy the application code
COPY . /app/

# Step 5: Expose the port Dash will run on
EXPOSE 80

# Step 6: Run the application using Gunicorn for production
# CMD ["python", "app.py"]
CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:server"]
