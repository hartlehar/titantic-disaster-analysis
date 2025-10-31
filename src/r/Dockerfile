FROM rocker/r-base:4.3.1

RUN mkdir src

# Copy R scripts
COPY r/ /src/

# Copy data
COPY data/ /src/data

# Set working directory
WORKDIR /src

# Install packages
RUN Rscript install_packages.R

CMD ["Rscript", "train.R"]