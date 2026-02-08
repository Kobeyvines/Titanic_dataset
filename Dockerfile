# 1. Use an official Python runtime as a parent image
# "slim" versions are lighter and faster to build
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy just the requirements first (Docker Cache optimization)
# This way, Docker won't re-install pandas every time you change a line of code.
COPY requirements.txt .

# 4. Install dependencies
# --no-cache-dir keeps the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code
COPY . .

# 6. Expose the port the app runs on
# (Render/Heroku/AWS usually provide a PORT env var, but 8000 is a good default)
EXPOSE 8000

# 7. Define the command to run your app
# We use 'sh -c' so we can use the ${PORT} variable injected by cloud providers
CMD ["sh", "-c", "uvicorn api.index:app --host 0.0.0.0 --port ${PORT:-8000}"]
