# Project Description

This repository contains a possible solution to detect fake users for a marketplace. 

The *tree* of this repository is:

````bash
.
├── Dockerfile
├── README.md
├── config.py
├── data
│   └── fake_users.csv
├── model
│   └── .gitkeep
├── notebooks
│   └── Data_Exploration.ipynb
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── inference.py
│   ├── training.py
│   └── utils.py
└── tests
    ├── conftest.py
    ├── test.csv
    └── test_utils.py
````

----

## Problem Context

A marketplace is being attacked by bots that produce fake clicks and leads. The
marketplace reputation might be affected if sellers get tons of fake leads and
receive spam from bots. On top of that, these bots introduce noise to our models
in production that rely on user behavioural data. We need to save the marketplace
reputation detecting these fake users. To do so, we have a dataset of logs of a
span of five minutes. Each entry contains the user id (`UserId`), the action that
a user made (`Event`), the category it interacted with (`Category`) and a column
(`Fake`) indicating if that user is fake (1 is fake, 0 is a real user).

An example of how the data looks like:

| UserId | Event      | Category  | Fake |
|--------|------------|-----------|------|
| XE321R | click_ad   | Motor     | 1    |
| ZE458P | send_email | Motor     | 0    |
| XE321R | click_ad   | Motor     | 1    |

----

## How to run the code

**Note:**
Before to run the code, please make sure that you put the data files (.csv) in the
data folder.

To run the code, we can use a *Docker container* since it can run in different OS
environments. For that, we have to build first the image using the Dockerfile:

```bash
docker build . -t adevinta_docker_image
```

Then, we have to run a docker container based on the former docker image:

```bash
docker run -d --name adevinta_test -v $(PWD):/adevinta -it adevinta_container
```

Finally, we can run the scripts to train a model and to get a inference for a given file:

```bash
# To train a model and considering that the input file is in the container 
docker exec -it adevinta_test python src/training.py --input-file data/fake_users.csv

# To infer for a given file
docker exec -it adevinta_test python src/inference.py --input-file data/fake_users.csv --output-file data/output.csv
```

You can recover the `output.csv` file in the data folder with the predictions.

----

## Next steps

* Improve the feature engineering for the model

