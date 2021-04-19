# -*- coding: utf-8 -*-
import read_data as rd
import pandas as pd

def clean_data(file):
    """
    Process the data.
    Data cleaning may include reduction or augmentation depending on how skewed the data is.
    """
    df = rd.read_data(file).drop_duplicates( subset = ['TITLE'] )
    dt = df[['CATEGORY', 'TITLE']].sample(50000, random_state = 22)
    cat_count = pd.DataFrame(dt.CATEGORY.value_counts())
    return (cat_count, dt)