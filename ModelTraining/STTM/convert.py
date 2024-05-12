import pandas as pd
"""
This File is used to convert any .csv files into the .txt format that
STTM supports. After this conversion, you can execute the STTM command
from command line on the newly created file.

Syntax:
java -jar jar/STTM.jar -model <topic_model> -corpus <data_file> -ntopics <n> -name <name>

e.g.
java -jar jar/STTM.jar -model LDA -corpus twitter.txt -ntopics 30 -name twitter                                      
"""

dataset_path = "DataCollection/Data/islamophobic-tweets-clean.csv"
output_dataset_name = "twitter-80.txt"

df = pd.read_csv(dataset_path)
df['document'].to_csv(output_dataset_name, index=False, header=False)
