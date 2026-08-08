import sys, json
import nbformat as nbf
from nbclient import NotebookClient

def build_and_run(cells, out_path, kernel="python3", execute=True, timeout=1800):
    """cells: list of (type, content) tuples, type in {'markdown','code'}"""
    nb = nbf.v4.new_notebook()
    nb["cells"] = []
    for typ, content in cells:
        if typ == "markdown":
            nb["cells"].append(nbf.v4.new_markdown_cell(content))
        else:
            nb["cells"].append(nbf.v4.new_code_cell(content))
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    if execute:
        client = NotebookClient(nb, timeout=timeout, kernel_name=kernel,
                                 resources={"metadata": {"path": "/home/claude/proyecto"}})
        client.execute()
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook guardado: {out_path}")
