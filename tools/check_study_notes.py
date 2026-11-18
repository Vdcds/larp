"""Check study-note links and examples against the repository programs."""

import argparse
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import re
from unittest.mock import patch
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def heading_ids(text):
    headings = re.findall(r"^#{1,6} (.+)$", text, re.MULTILINE)
    return {
        re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        for heading in headings
    }


def check_documents():
    documents = [ROOT / "README.md", ROOT / "CHEATSHEET.md"]
    documents += sorted((ROOT / "docs").glob("*.md"))
    assert len(documents) == 14, "Expected README, cheat sheet, run guide, and 11 explainers"
    for document in documents:
        text = document.read_text()
        assert text.count("\n```") % 2 == 0, f"Unclosed code fence in {document}"
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                continue
            target = document.parent / unquote(parsed.path) if parsed.path else document
            assert target.is_file(), f"Broken link in {document}: {link}"
            if parsed.fragment:
                assert unquote(parsed.fragment) in heading_ids(target.read_text()), (
                    f"Missing heading in {document}: {link}"
                )
        for snippet in re.findall(r"```python\n(.*?)\n```", text, re.DOTALL):
            ast.parse(snippet, filename=str(document))
    for number in range(1, 12):
        text = (ROOT / f"pr{number}.txt").read_text()
        ast.parse(text, filename=f"pr{number}.txt")
        notebook = json.loads((ROOT / f"pr{number}.ipynb").read_text())
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        ast.parse(code, filename=f"pr{number}.ipynb")
        if number in (4, 7):
            assert ast.dump(ast.parse(text)) == ast.dump(ast.parse(code)), (
                f"PR{number} text and notebook code differ"
            )
    print("PASS: 14 documents, local links, Python snippets, and 11 program pairs")


def run_program(number, notebook=False, output=None):
    if notebook:
        data = json.loads((ROOT / f"pr{number}.ipynb").read_text())
        code = "\n".join(
            "".join(cell["source"])
            for cell in data["cells"]
            if cell["cell_type"] == "code"
        )
    else:
        code = (ROOT / f"pr{number}.txt").read_text()
    namespace = {"__name__": "__main__"}
    with redirect_stdout(output if output is not None else io.StringIO()):
        exec(compile(code, f"pr{number}", "exec"), namespace)
    return namespace


def check_search_and_rules():
    for temperature, motion, ac, light in [
        (28, "no", "OFF", "OFF"),
        (29, "yes", "ON", "ON"),
        (30, "no", "ON", "OFF"),
        (20, "yes", "OFF", "ON"),
    ]:
        with patch("random.randint", return_value=temperature):
            with patch("random.choice", return_value=motion):
                result = run_program(1)
        assert (result["ac"], result["light"]) == (ac, light)
    result = run_program(2)
    expected = ["A", "B", "F", "J", "L"]
    assert result["bfs_path"] == result["dfs_path"] == expected
    assert len(expected) - 1 == 4
    result = run_program(3)
    assert result["greedy_path"] == ["A", "C", "F", "G"]
    assert result["astar_path"] == ["A", "B", "D", "G"]
    assert result["total_cost"] == 6
    graph, heuristic = result["graph"], result["heuristic"]
    path = result["greedy_path"]
    assert sum(dict(graph[u])[v] for u, v in zip(path, path[1:])) == 9
    for node, edges in graph.items():
        for neighbor, cost in edges:
            assert heuristic[node] <= cost + heuristic[neighbor]
    print("PASS: PR1 threshold examples, PR2 routes, PR3 costs and heuristic")


def check_expert_system():
    scenarios = [
        (["yes"] * 7, ["flu", "cold", "infection", "doctor_visit"]),
        (["no"] * 7, []),
        (["yes", "yes", "yes", "no", "no", "no", "no"], ["flu", "infection"]),
        (["no", "yes", "no", "yes", "yes", "no", "no"], ["cold"]),
        (["no", "no", "no", "no", "no", "yes", "yes"], ["doctor_visit"]),
        (["yes", "no", "no", "no", "no", "no", "no"], []),
        (["YES", "Yes", "yes", "no", "no", "no", "no"], ["flu", "infection"]),
        ([" yes "] * 7, []),
    ]
    for notebook in (False, True):
        for answers, expected in scenarios:
            output = io.StringIO()
            with patch("builtins.input", side_effect=answers) as prompt:
                result = run_program(4, notebook=notebook, output=output)
            assert prompt.call_count == 7
            assert result["conclusions"] == expected
            labels = re.findall(r"^Possible condition: (.+)$", output.getvalue(), re.MULTILINE)
            assert labels == [label.replace("_", " ").title() for label in expected]
            assert ("No matching condition found." in output.getvalue()) == (not expected)
    print("PASS: PR4 text and notebook with 8 interactive answer scenarios each")


def check_ml_programs():
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib.pyplot as plt
    import numpy as np

    plt.show = lambda: plt.close("all")
    for notebook in (False, True):
        result = run_program(7, notebook=notebook)
        assert len(result["X_train"]) == 455
        assert len(result["X_test"]) == 114
        assert np.allclose(result["scaler"].mean_, result["X_train"].mean(axis=0))
        assert len(result["lr_pred"]) == len(result["knn_pred"]) == 114
    print("PASS: PR7 text and notebook run independently with a training-only scaler")
    result = run_program(8)
    assert result["cm"].shape == (3, 3) and result["cm"].sum() == 30
    result = run_program(9)
    assert result["X_scaled"].shape == (150, 4)
    assert len(result["wcss"]) == 10
    assert result["df"]["KMeans_Cluster"].nunique() == 3
    assert result["df"]["Hierarchical_Cluster"].nunique() == 3
    result = run_program(10)
    assert result["X"].shape[1] == 64
    pca = result["pca_model"].named_steps["pca"]
    assert 0 < pca.n_components_ < 64
    assert pca.explained_variance_ratio_.sum() >= 0.95
    assert result["X_2d"].shape[1] == 2
    result = run_program(11)
    searches = [result["svm_grid"], result["rf_grid"], result["ada_grid"]]
    assert [len(search.cv_results_["params"]) for search in searches] == [6, 6, 4]
    assert all(search.n_splits_ == 5 and search.refit for search in searches)
    print("PASS: PR8 matrix, PR9 fitted features and clusters, PR10 PCA, PR11 grid counts")
    print("PR5 and PR6 were not run; they require an external CSV or a possible download.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml", action="store_true", help="also run PR7 through PR11")
    args = parser.parse_args()
    check_documents()
    check_search_and_rules()
    check_expert_system()
    if args.ml:
        check_ml_programs()
