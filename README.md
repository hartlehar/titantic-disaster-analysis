# Titanic Disaster Data Analysis

Data was taken from its respective [Kaggle competition](https://www.kaggle.com/competitions/titanic/data). 

For the code to run properly, please download the dataset from the competition and save the `.csv` files into `/src/data/` (you may need to create `/src/data/`).

Data transformation and modeling was done in Python and R. Their respecitve scripts are located in `/src/py/` for Python and `/src/r/` for R.

To run the code, please use Docker to build the images and run them. You will need to be in the `src` directory and specify which `Dockerfile` (Python or R version) you want to build.

### Example:
```
docker build -f [language]/Dockerfile -t [image-name] .
```

Fill `[language]` with the desired programming language (`py` for Python or `r` for R), and `[image-name]` with the desired image name.
