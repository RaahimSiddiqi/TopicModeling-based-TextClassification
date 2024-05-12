import pandas as pd

def get_topic_assignments():
    assignments_file_path = "STTM/results/model.topicAssignments"
    word_topic_assignments = []
    with open(assignments_file_path, 'r') as f:
        for line in f:
            topics = list(map(int, line.strip().split()))
            word_topic_assignments.append(topics)
    return word_topic_assignments


def get_words_set(documents):
    words = set()
    assigments = get_topic_assignments()

    for index in documents.index:
        assigment = assigments[index]
        document = documents[index].split(' ')
        unique_words = set(document)
        words.update(unique_words)
        # print(len(assigment), len(document))
    return words


df = pd.read_csv("DataCollection/Data/train-clean.csv")[:10]
set0 = get_words_set(df[df['class']==0]['document'])
set1 = get_words_set(df[df['class']==1]['document'])