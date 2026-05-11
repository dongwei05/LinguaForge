"""Move the Modelfile section ahead of the bench section in the GGUF notebook.

So that even if the (memory-pressured) bench cell crashes the Kaggle kernel,
the Modelfile + GGUF are already persisted to /kaggle/working before that.
"""
import json
import sys

NB = "notebooks/auto_run_gguf/linguaforge_gguf.ipynb"
nb = json.loads(open(NB, encoding="utf-8").read())
cells = nb["cells"]

# Find indices by leading markdown header text.
def find(prefix: str) -> int:
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown":
            src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            if src.strip().startswith(prefix):
                return i
    sys.exit(f"could not find header starting with {prefix!r}")


i_bench = find("## 5. CPU benchmark")
i_model = find("## 6. Ollama Modelfile")
i_clean_md = i_model + 2  # markdown header for cleanup or the cleanup cell itself
# Each section = markdown header + code cell.
bench_block = cells[i_bench : i_bench + 2]
model_block = cells[i_model : i_model + 2]
# Final ordering: ...quantize... | modelfile | bench | cleanup...
new_cells = cells[:i_bench] + model_block + bench_block + cells[i_model + 2 :]
nb["cells"] = new_cells
open(NB, "w", encoding="utf-8").write(json.dumps(nb, indent=1, ensure_ascii=False))

print("reordered. new order:")
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    head = src.split("\n")[0][:80]
    print(f"  {i:2d} {c['cell_type']:8s} {head}")
