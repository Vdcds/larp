# AIML practical revision

Two-minute read-aloud explainers and a [quick cheat sheet](CHEATSHEET.md) for all 11 programs. Each explainer follows the code in this repository and points out the assumptions you may need to explain in a viva.

## Pick a practical

| Practical | Topic | Two-minute explainer | Program | Notebook |
| --- | --- | --- | --- | --- |
| PR1 | Intelligent agent and PEAS | [Read](docs/pr01.md) | [Code](pr1.txt) | [Notebook](pr1.ipynb) |
| PR2 | BFS and DFS | [Read](docs/pr02.md) | [Code](pr2.txt) | [Notebook](pr2.ipynb) |
| PR3 | Greedy best-first search and A* | [Read](docs/pr03.md) | [Code](pr3.txt) | [Notebook](pr3.ipynb) |
| PR4 | Forward and backward chaining | [Read](docs/pr04.md) | [Code](pr4.txt) | [Notebook](pr4.ipynb) |
| PR5 | Titanic preprocessing and EDA | [Read](docs/pr05.md) | [Code](pr5.txt) | [Notebook](pr5.ipynb) |
| PR6 | Linear regression | [Read](docs/pr06.md) | [Code](pr6.txt) | [Notebook](pr6.ipynb) |
| PR7 | Logistic regression and KNN | [Read](docs/pr07.md) | [Code](pr7.txt) | [Notebook](pr7.ipynb) |
| PR8 | Gaussian Naive Bayes | [Read](docs/pr08.md) | [Code](pr8.txt) | [Notebook](pr8.ipynb) |
| PR9 | K-means and hierarchical clustering | [Read](docs/pr09.md) | [Code](pr9.txt) | [Notebook](pr9.ipynb) |
| PR10 | PCA with logistic regression | [Read](docs/pr10.md) | [Code](pr10.txt) | [Notebook](pr10.ipynb) |
| PR11 | Model comparison and grid search | [Read](docs/pr11.md) | [Code](pr11.txt) | [Notebook](pr11.ipynb) |

## Before the practical

The [run guide](docs/running.md) covers installation, the missing Titanic CSV, and common errors. The cheat sheet has code reminders, formulas, and short viva answers. The explainers are scripts intended for roughly two minutes at a calm speaking pace; they are text, not audio or video.

PR7 now creates its train/test split before scaling in both file formats. PR5 still needs `Titanic-Dataset.csv`, which is not included. Its explainer describes the preprocessing leakage caveat in the original program.

The notes describe these teaching examples. Model scores depend on the dataset, split, and library version, so the notes do not promise a fixed accuracy.
