import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import re

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_files_from_gdrive(url: str, fname: str) -> None:
    file_id = url.split("/")[5]
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, fname, quiet=False)
    
def clean(df, col):
    """Cleaning Twiitter data
    
    Arguments:
        df {[pandas dataframe]} -- Dataset that needs to be cleaned
        col {[string]} -- column in which text is present
    
    Returns:
        [pandas dataframe] -- Datframe with a "clean_text" column
    """    
    df["clean_text"] = df[col]
    df["clean_text"] = (
        (df["clean_text"])
            .apply(lambda text: re.sub(r"RT\s@\w+:", "Retweet", text)) #Removes RTS
            .apply(lambda text: re.sub(r"@", "mention ", text)) # Replaces @ with mention
            .apply(lambda text: re.sub(r"#", "hashtag ", text)) # Replaces # with hastag
            .apply(lambda text: re.sub(r"http\S+", "", text)) # Removes URL
        )
    return df
