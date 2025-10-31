# Titanic Disaster Data Analysis

Data was taken from its respective [Kaggle competition](https://www.kaggle.com/competitions/titanic/data). 

For the code to run properly, please download the dataset from the competition and save the `.csv` files into `/src/data/` (you may need to create `/src/data/`).

Data transformation and modeling was done in Python and R. Their respective scripts are located in `/src/py/` for Python and `/src/r/` for R.

## Running the Code

To run the code, you can either use Docker or directly from the terminal. 

### Using Docker

To build the images, you first need to be in the `src` directory.

Then, specify which `Dockerfile` (Python or R version) you want to build.

You can use the following command to build the image:
```
docker build -f [language]/Dockerfile -t [image-name] .
```

Replace `[language]` with the desired programming language (`py` for Python or `r` for R), and `[image-name]` with the desired image name.

Then, you should be able to run the image from the Docker Desktop app, or you can run it from the terminal using:
```
docker run [image-name]
```

### From the Terminal

You will also need to be in the `src` directory to run the scripts.

Please make sure that you have installed the required packages specified in `/py/requirements.txt` for Python (using `pip` or `conda`) and `r/install_packages.R` for R (using `Rscript`) before running.

You can use the following commands to run the scripts:

#### For Python
```
python ./py/train.py
```

#### For R
```
Rscript ./r/train.R
```

Running either script will create `test_predictions.csv` in the `src` directory, which contains the predictions from the models.

`test_predictions.csv` is formatted into 2 columns:

  - PassengerId
  - Survived - 0 = No, 1 = Yes
