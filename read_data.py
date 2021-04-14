# -*- coding: utf-8 -*-

import pandas as pd


def read_data(file):
    """
    Read simulation data.
    Data is a csv formatted file.
    """
    data = pd.read_csv(file)

    return data

