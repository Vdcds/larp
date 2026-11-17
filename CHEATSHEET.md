# AIML quick cheat sheet

These are revision notes for the programs in this repo. Code fragments show key lines and assume the surrounding program's variables and imports. Full explanations and source links are in the [practical index](README.md#pick-a-practical).

## PR1: Intelligent agent and PEAS

PEAS means Performance measure, Environment, Actuators, and Sensors. Here those are comfort and energy saving, a smart home, AC and lights, and temperature and motion.

```python
ac = "ON" if temperature > 28 else "OFF"
light = "ON" if motion == "yes" else "OFF"
```

- Temperature 30 and motion `no` give AC `ON` and light `OFF`.
- Temperature 28 keeps the AC off. The comparison is `>`, not `>=`.
- The readings are random simulations. The outputs do not control hardware.
- Viva: Why is it a reflex agent? Its current reading selects a fixed action rule, without memory or learning.

## PR2: BFS and DFS

| Detail | BFS | DFS |
| --- | --- | --- |
| Frontier | FIFO queue | LIFO stack |
| Removal | `queue.popleft()` | `stack.pop()` |
| Search order | Shallower paths first | One branch first |
| Shortest-route guarantee | By edge count for unweighted graphs | No general guarantee |

```python
node = path[-1]
new_path = path + [neighbour]
route_length = len(path) - 1
```

Both return `A -> B -> F -> J -> L`, length `4`, on this graph. `reversed(graph[node])` in DFS preserves left-to-right neighbor exploration with a stack. `visited` prevents repeated expansion. Textbook node-based BFS and DFS take O(V + E) traversal time with adjacency lists; this implementation also copies whole paths and can queue duplicate destinations.

Viva: Can BFS solve minimum-weight routing? Its shortest-path guarantee requires equal edge costs. It does not rank different weights.

## PR3: Greedy best-first search and A*

| Method | Priority | Supplied graph's result |
| --- | --- | --- |
| Greedy | `h(n)` | `A -> C -> F -> G`, cost `9` |
| A* | `g(n) + h(n)` | `A -> B -> D -> G`, cost `6` |

```python
new_g = g + cost
new_f = new_g + heuristic[neighbour]
```

`g` is the cost already paid; `h` estimates the cost remaining. An admissible heuristic never overestimates the true cheapest remaining cost. Consistency requires `h(u) <= cost(u, v) + h(v)` on every edge. The supplied heuristic is consistent. With this implementation's permanent `visited` set, consistency is the relevant condition for safely closing nodes without reopening them.

Viva: Why can greedy miss the cheapest route? It ignores the cost already paid. The name `greedy_bfs` here means best-first, not breadth-first.

## PR4: Forward and backward chaining

| Method | Starts with | Main question |
| --- | --- | --- |
| Forward chaining | Facts | What conclusions follow? |
| Backward chaining | A goal | Can the supporting conditions be proved? |

```text
fever AND cough => flu
fever AND cold  => infection
flu            => doctor_visit
```

All conditions use AND through `all(...)`. Forward chaining derives `flu`, `infection`, and `doctor_visit` and stops when a scan adds nothing. Backward chaining proves `doctor_visit` through `flu`, then `fever` and `cough`.

Viva: What is missing for arbitrary rule graphs? The backward function needs cycle handling to avoid infinite recursion. The dictionary stores one condition list per conclusion; it does not represent alternative rules for the same conclusion.

## PR5: Titanic preprocessing and EDA

The sequence is inspect, impute, remove duplicates, drop columns, encode, split, scale, and explore. The CSV must be supplied separately.

```python
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train[columns] = scaler.fit_transform(X_train[columns].astype(float))
X_test[columns] = scaler.transform(X_test[columns].astype(float))
```

- `Age` and `Fare` use median imputation; `Embarked` uses the mode.
- `Sex` maps `male` to `0` and `female` to `1`; `Embarked` becomes indicator columns.
- `z = (x - training_mean) / training_std`. This is standardization, not min-max scaling.
- Histograms show distributions; boxplots summarize spread and possible outliers.
- The code produces prepared data and plots, not a trained classifier.

Viva: Does this program avoid all preprocessing leakage? No. The scaler fits only training data, but medians and the mode are computed before splitting. For evaluation, split before learning those statistics.

## PR6: Linear regression

```python
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Prediction is `b0 + sum(bi * xi)`. The coefficients minimize squared training residuals.

| Metric | Meaning | Direction |
| --- | --- | --- |
| MAE | Mean absolute error | Lower is better |
| MSE | Mean squared error | Lower is better |
| RMSE | Square root of MSE | Lower is better |
| R-squared | `1 - SSE/SST` relative to the target mean | Higher is better; negative is possible |

California Housing targets are median values for census block groups, in units of $100,000. A value of `2.5` corresponds to about $250,000 in the dataset. New rows must have the same features in the same order.

Viva: Is R-squared an accuracy percentage? No. It is a regression score relative to a mean-prediction baseline.

## PR7: Logistic regression and KNN

The update adds the missing split before the existing scaler:

```python
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)
```

- Logistic regression learns a linear score and uses the sigmoid for a binary probability.
- KNN with `k=3` votes among the three nearest training examples.
- Fit the scaler on training features, then transform test features with that scaler.
- Dataset labels are `0 = malignant`, `1 = benign`.
- Confusion-matrix rows are actual classes; columns are predicted classes.
- Precision is `TP / (TP + FP)`. Recall is `TP / (TP + FN)`. F1 is their harmonic mean.

Viva: Why scale for KNN? Distances otherwise give larger-scale features too much influence. When discussing recall for malignant cases, treat malignant as the class of interest rather than assuming numeric label one is the positive class.

## PR8: Gaussian Naive Bayes

```python
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

The class score is proportional to `P(class) * product(P(feature_i | class))`. GaussianNB uses a Gaussian density for each feature within each class. Its "naive" assumption is conditional independence given the class.

Iris has four input measurements and three species labels. `data.target_names` provides their names. The code uses an 80/20 split with seed 42 and no `stratify` argument. Diagonal confusion-matrix cells are correct predictions; off-diagonal cells are confusions.

Viva: Why GaussianNB? The inputs are continuous measurements. Multinomial and Bernoulli Naive Bayes make different assumptions about their features.

## PR9: K-means and hierarchical clustering

```python
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
hierarchical = AgglomerativeClustering(n_clusters=3, linkage="ward")
```

- Both fit the four standardized Iris features without training on species labels.
- K-means alternates assignments and centroid updates. `inertia_` is the sum of squared distances to assigned centroids.
- The elbow loop tries `k=1` through `10`; the final `k=3` is hard-coded, not automatically selected.
- `n_init=10` tries ten initializations. It is not the number of update iterations.
- Ward clustering merges groups with the smallest increase in within-cluster variance.
- The dendrogram shows the merge hierarchy. Scatter plots show only two of the four fitted features.

Viva: Is cluster zero the first species? No. Cluster IDs are arbitrary, and IDs from two clustering algorithms are not automatically aligned.

## PR10: PCA

```python
pca_model = make_pipeline(
	StandardScaler(),
	PCA(n_components=0.95),
	LogisticRegression(max_iter=3000)
)
```

Digits starts with 64 pixel features. PCA creates new perpendicular component directions from combinations of the inputs. It uses variance, without using class labels.

- `0.95` sets the retained-variance target; it is neither 95 components nor an accuracy target.
- The fitted count is `pca_model.named_steps["pca"].n_components_`.
- PCA centers inputs but does not replace the separate standardization step.
- The training pipeline fits preprocessing only on training data.
- A different `PCA(n_components=2)` produces the visualization.

Viva: Does PCA always improve accuracy? No. It may remove noise or discard useful class information. Compare held-out results without assuming improvement.

## PR11: Model optimization

| Model | Grid | Combinations |
| --- | --- | --- |
| SVM | `C = [0.1, 1, 10]`, kernel `linear` or `rbf` | 6 |
| Random Forest | `n_estimators = [50, 100]`, `max_depth = [None, 5, 10]` | 6 |
| AdaBoost | `n_estimators = [50, 100]`, `learning_rate = [0.5, 1.0]` | 4 |

Five-fold grid search evaluates 16 combinations across five folds: 80 candidate-fold fits, plus the default final refit for each search. Baseline fitting and the separate `cross_val_score` loops are additional work.

- SVM uses a scaled pipeline; its grid keys are `model__C` and `model__kernel`.
- `best_params_` stores the chosen settings.
- `best_score_` is mean training-CV accuracy, not held-out test accuracy.
- `best_estimator_` is refitted on all training data by default.
- Tuning can leave test accuracy unchanged or make it worse.

Viva: Why keep a test set outside grid search? The held-out set estimates performance on examples that did not choose the settings. Repeatedly selecting a model from test results weakens that separation.
