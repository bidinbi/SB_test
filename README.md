Тестовое задание в команду СберСпасибо  
==============================
При выполнении задания я выделил мало времени на качественное решение самой задачи, т.к. главная задача была показать навыки кода.  
Использовался шаблон cookiecutter. 

В папке data/raw лежат сырые данные.  

В папке src/models лежат три файла:  
    1) parent_model.py -- создается родительский класс модели, от которой будут наследоваться следующие модели  
    2) popular_model.py -- создается простенькая рекомендационная модель, которая предсказывает всегда самые популярные товары  
    3) svd_model.py -- создается модель рекомендаций на основе сингулярного разложения  
    
Файл run.py:  
    В этом файле происходит логика запуска эксперимента. Создается, обрабатывается и разбивается на трейн и тест датасет, обучаются и тестируются различные модели, результаты выводятся в терминал и сохраняются в виде pandas dataframe в папке reports/experiments. При запуске файл принимает один аргумент -- название эксперимента. Пример запуска из командной строки: ./run.py test_exp


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    |
    └── run.py             <- script to run


--------
