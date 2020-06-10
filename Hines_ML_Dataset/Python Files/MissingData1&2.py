import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

MD1 = pd.read_csv('Datasets/MissingData1.txt', sep="\t", header=None)
MD1.interpolate(axis=0, limit_direction="both")

MD2 = pd.read_csv('Datasets/MissingData2.txt', sep="\t", header=None)
MD2.interpolate(axis=0, limit_direction="both")