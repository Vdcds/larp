# Run the practicals

## Set up Python

From the repository folder, run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyterlab
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Run a program

The `.txt` files contain Python code. Python can execute them directly:

```sh
python pr1.txt
python pr7.txt
```

For a notebook, start JupyterLab:

```sh
jupyter lab
```

Open the required `.ipynb`, select the environment's kernel, and run all cells from top to bottom. To check that a notebook does not depend on old variables, restart its kernel before running all cells.

## Supply PR5's dataset

Place your course's `Titanic-Dataset.csv` beside `pr5.txt`. Start Python or Jupyter from that folder. The program expects these columns:

```text
PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch,
Ticket, Fare, Cabin, Embarked
```

Use the course dataset's exact column names and values. The encoding expects `male` and `female` in `Sex`. The CSV is not included in this repository.

## Load PR6's dataset

Connect to the internet for the first call to `fetch_california_housing()`. Later runs can use scikit-learn's local dataset cache. If the download fails, check the connection and retry.

## Resolve common errors

| Error or symptom | Action |
| --- | --- |
| `ModuleNotFoundError` | Install the packages in the interpreter or notebook kernel that runs the program. |
| `FileNotFoundError: Titanic-Dataset.csv` | Put the CSV in the working directory or edit `pd.read_csv(...)` to use its real path. |
| PR7 reports `X_train` is undefined | Pull the latest repo, restart the kernel, and run all cells. The update adds the missing split. |
| A plot pauses a script | Close the plot window to continue to the next plot. |
| Scores differ from a friend's machine | Compare dataset files, splits, and library versions before comparing results. |

## Reproduce the study-note check

The standard-library check verifies the local document links and PR1–PR4 examples:

```sh
python tools/check_study_notes.py
```

To also run PR7–PR11 in fresh namespaces with the installed ML packages:

```sh
python tools/check_study_notes.py --ml
```

The ML check uses the noninteractive Matplotlib backend. It does not run PR5, which needs the external CSV, or PR6, which may need a dataset download.
