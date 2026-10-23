"""Build one experiment: src/expNN.py -> notebook (executed) -> PDF via pandoc template.

usage: python tools/build.py N [--no-exec]
"""
import base64, os, re, shutil, subprocess, sys, json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
N = int(sys.argv[1])
NOEXEC = "--no-exec" in sys.argv
NAMES = json.loads((ROOT / "tools" / "experiments.json").read_text())
meta = NAMES[str(N)]
src = ROOT / "src" / f"exp{N:02d}.py"
work = ROOT / "_build" / f"exp{N:02d}"
out_dir = ROOT / f"Experiment {N}"
work.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(exist_ok=True)
nb_path = out_dir / f"Experiment_{N}.ipynb"

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def parse(text):
    """Split percent-format script into (kind, title, body) cells."""
    cells, cur = [], None
    for line in text.splitlines():
        m = re.match(r"# %%(.*)$", line)
        if m:
            if cur: cells.append(cur)
            hdr = m.group(1).strip()
            if hdr.startswith("[markdown]"):
                cur = ["markdown", None, []]
            else:
                t = re.match(r"title:\s*(.*)", hdr)
                cur = ["code", t.group(1).strip() if t else None, []]
        elif cur is not None:
            cur[2].append(line)
    if cur: cells.append(cur)
    out = []
    for kind, title, body in cells:
        while body and not body[-1].strip(): body.pop()
        while body and not body[0].strip(): body.pop(0)
        if kind == "markdown":
            body = [re.sub(r"^# ?", "", l) for l in body]
        out.append((kind, title, "\n".join(body)))
    return out


def build_nb():
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, title, body in parse(src.read_text()):
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(body))
        else:
            c = nbformat.v4.new_code_cell(body)
            if title: c.metadata["title"] = title
            nb.cells.append(c)
    return nb


if NOEXEC and nb_path.exists():
    nb = nbformat.read(nb_path, as_version=4)
else:
    nb = build_nb()
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", TF_ENABLE_ONEDNN_OPTS="0", PYTHONHASHSEED="0")
    os.environ.update(env)
    client = NotebookClient(nb, timeout=7200, kernel_name="python3",
                            resources={"metadata": {"path": str(work)}})
    client.execute()
    nbformat.write(nb, nb_path)
    print("notebook executed ->", nb_path)

# ---- collect code + outputs -------------------------------------------------
code_cells = [c for c in nb.cells if c.cell_type == "code"]
(work / "experiment_code.py").write_text(
    "\n\n".join(c.source for c in code_cells).encode("ascii", "replace").decode() + "\n")


def clean_text(s):
    s = ANSI.sub("", s)
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[^\n]*\r(?=[^\n])", "", s)  # keep last carriage-return state
    s = re.sub("[\u2500-\u257f]", lambda m: "|" if m.group() in "\u2502\u2503\u2551" else "-" if m.group() in "\u2500\u2501\u2550" else "+", s)
    s = s.encode("ascii", "replace").decode().replace("?�", "?")
    lines = [l.rstrip() for l in s.split("\n") if "Downloading data" not in l and "?????" not in l]
    if len(lines) > 70:
        lines = lines[:40] + [f"... ({len(lines) - 55} lines omitted) ..."] + lines[-15:]
    return "\n".join(lines).strip("\n")


out_md, figno = [], 0
for c in code_cells:
    title = c.metadata.get("title")
    if not title: continue
    texts, imgs = [], []
    for o in c.outputs:
        if o.output_type == "stream" and o.name == "stdout":
            texts.append(o.text)
        elif o.output_type in ("execute_result", "display_data"):
            d = o.data
            if "image/png" in d:
                imgs.append(base64.b64decode(d["image/png"]))
            elif "text/plain" in d and o.output_type == "execute_result":
                texts.append(d["text/plain"] + "\n")
    if not texts and not imgs: continue
    out_md.append(f"## {title}\n")
    txt = clean_text("".join(texts))
    if txt:
        out_md.append("```{=latex}\n\\begin{lstlisting}[numbers=none,language={},xleftmargin=0pt,framexleftmargin=0pt,"
                      "basicstyle=\\fontsize{8}{9.6}\\selectfont\\ttfamily]\n" + txt + "\n\\end{lstlisting}\n```\n")
    for im in imgs:
        figno += 1
        fn = f"fig{figno:02d}.png"
        (work / fn).write_bytes(im)
        out_md.append(f"![{title}]({fn})\n")

doc = (ROOT / "docs" / f"exp{N:02d}.md").read_text()
doc = doc.replace("@@OUTPUT@@", "\n".join(out_md))
head = (f"---\ntitle: \"Experiment {N}: {meta['title']}\"\nlab: \"Experiment {N}\"\n"
        f"subject: \"Generative AI Laboratory ({meta['co']})\"\ndate: \"22 October 2026\"\ncode: experiment_code.py\nlanguage: Python\n---\n\n")
(work / "doc.md").write_text(head + doc)

pdf = out_dir / f"Experiment_{N}_Implementation.pdf"
cmd = ["pandoc", "doc.md", "-f", "markdown-implicit_figures+raw_tex", "--template", str(ROOT / "template/lab-template.tex"),
       "--lua-filter", str(ROOT / "template/code-filter.lua"), "--pdf-engine=pdflatex",
       "-V", "geometry:margin=2.3cm", "-o", str(pdf)]
pdf.unlink(missing_ok=True)   # never report a stale PDF as success
r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-3000:])
print("PDF ->", pdf if (r.returncode == 0 and pdf.exists()) else "FAILED")
