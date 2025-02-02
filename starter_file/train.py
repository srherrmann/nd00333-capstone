import argparse
from azureml.core import Run, Dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=50, help="Number of decision trees to train for the randomf forest")
    parser.add_argument('--max_depth', type=int, default=100, help="Maximum tree depth")
    parser.add_argument('--min_samples_split', type=int, default=2, help="Minimum number of samples required to split an internal node")
    parser.add_argument('--min_samples_leaf', type=int, default=1, help="Minimum number of samples required to be at a leaf node")
    parser.add_argument('--max_features', type=str, default='sqrt', help="Number of features to consider when looking for the best split")

    args = parser.parse_args()

    # convert -1 to None:
    for param in ['max_depth', 'min_samples_split', 'min_samples_leaf']:
        if getattr(args, param) == -1:
            setattr(args, param, None)

    run = Run.get_context()
    # log each argument
    for key, value in vars(args).items():
        run.log(key, value)

    ws = run.experiment.workspace
    dataset_name = "heart-failure-clinical-data"
    ds = Dataset.get_by_name(ws, name=dataset_name)

    # TODO: Split data into train and test sets.
    df = ds.to_pandas_dataframe()

    # Remove records with NaN or inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    x = df.drop(["DEATH_EVENT"], axis=1)
    y = df["DEATH_EVENT"]

    ### YOUR CODE HERE ###
    x_train, x_test, y_train, y_test = train_test_split(x,y)

    # DEBUG: print the unique values of x train and y train
    print("Unique values in x_train:")
    for column in x_train.columns:
        print(f"{column}: {x_train[column].unique()}")

    print("Unique values in y_train:")
    print(y_train.unique())

    model = RandomForestClassifier(n_estimators=args.n_estimators,
                                   max_depth=args.max_depth,
                                   min_samples_split=args.min_samples_split,
                                   min_samples_leaf=args.min_samples_leaf,
                                   max_features=args.max_features)\
                                   .fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    # Save the trained model
    model_path = "outputs/model.pkl"
    joblib.dump(model, model_path)

if __name__ == '__main__':
    main()