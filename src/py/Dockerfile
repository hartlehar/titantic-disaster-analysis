FROM python:3.11

RUN mkdir src

# Copy Python scripts
COPY py/ /src/

# Copy data
COPY data/ /src/data

# Set working directory
WORKDIR /src

# Install packages
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "train.py"]