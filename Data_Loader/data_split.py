from sklearn.model_selection import train_test_split

#Splits dataset (75% train 25% test)
def data_split(x, y, test_size=0.25, random_state=42):
    #Split preprocessed features and target into training and test sets
    return train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )