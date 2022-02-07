import pandas as pd
import os
from zipfile import ZipFile
from ensemble import Ensemble
import torch
from tqdm import tqdm


# Create submission for Competition
def create_submission(pred, zip=True):
    # Prediction to string
    pred = '\n'.join(pred)

    # Create label text
    file = open('labels.txt', 'w')
    file.write(pred)
    file.close()

    if zip:
        # Zip the file and delete the text
        ZipFile('../submission.zip', mode='w').write('labels.txt')
        os.remove('labels.txt')

if __name__ == '__main__':
    # Everything on to gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Importing data
    df = pd.read_csv('../datasets/preprocessed/dev.csv')

    # Load model
    model = Ensemble(models=['bert', 'roberta', 'xlnet', 'distilbert', 'albert', 'bertweet', 'twitterroberta'])
    model.to(device)

    # Get predictions
    uncertainty = False
    preds = [' '.join([str(x) for x in model(str(s), output_uncertainty=uncertainty)]) for s in tqdm(df['content'].tolist())]

    # Create submission
    create_submission(preds, zip=True)