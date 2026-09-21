# Putting this on GitHub

Read this once, then delete it (or keep it — it does no harm).

---

## Before anything else: ask your PI

The recordings are unpublished. Ask whether the repository can be public, and
when. **Create it Private and flip it to Public later** — private → public is
one click, public → private does not un-share anything people already copied.

Figures made from real data are usually fine to show even when the data is
not, but confirm which ones.

---

## Step 1 — Fill in the placeholders

Search the repository for `<your-username>` and `<your-email>` and replace
them. They appear in:

- `README.md` — the clone command, the citation `url`, the Contact section
- `docs/GITHUB_SETUP.md` — this file

Then decide the repository name. The code assumes `stamp`; if you pick
something else, change it in the same places.

---

## Step 2 — Add a license

`README.md` has a `## License` placeholder. Lab code is usually MIT (maximally
permissive) or BSD-3-Clause. **Ask your PI which one** — some institutions have
a policy. Once decided, GitHub can add the `LICENSE` file for you:
*Add file → Create new file → type `LICENSE` → "Choose a license template"*.

---

## Step 3 — Create the repository

On github.com: **New repository**.

- Name: `stamp`
- Description: *Self-supervised spatio-temporal transformer for cortical
  cholinergic dynamics across the lifespan in a mouse model of AD*
- **Private**
- Do **not** tick "Add a README" — this folder already has one.

---

## Step 4 — Push

```bash
cd /path/to/stamp

git init
git add .
git status          # ← READ THIS. See step 5 before continuing.
git commit -m "Initial commit: STAMP spatio-temporal masked causal transformer"
git branch -M main
git remote add origin https://github.com/<your-username>/stamp.git
git push -u origin main
```

---

## Step 5 — The one step not to skip

Before the first commit, read the output of `git status` and confirm that
**no `.h5`, `.mat`, `.npz`, `.pt` or `runs/` entry appears**.

`.gitignore` already excludes them, but a file that was staged before the
ignore rule existed stays staged. GitHub rejects any single file over 100 MB,
and — worse — a large file committed once remains in the history forever even
after you delete it, which is genuinely annoying to clean up.

If something slipped in:

```bash
git rm --cached path/to/the/file
```

Quick check:

```bash
git ls-files | grep -E '\.(h5|mat|npz|pt|pth|ckpt)$'    # must print nothing
du -sh .git                                            # should be a few MB
```

---

## Step 6 — Add the figures the README references

The README has no images yet. Adding three or four is the single highest-value
thing you can do to it — a reader decides whether a repository is serious in
about five seconds, and figures are what they look at.

```bash
cp runs/demo/figures/region_to_region_rollout.png docs/figures/
cp runs/demo/figures/brain_r2_by_age_genotype.png docs/figures/
cp runs/demo/figures/training_curves.png docs/figures/
```

Then reference them in `README.md`, for example under **Analysis**:

```markdown
![Region-to-region rollout](docs/figures/region_to_region_rollout.png)

*Left: all temporal lags pooled. Middle: lag 0 only — within one 1.5 s patch,
each region attends almost exclusively to itself. Right: lag ≥ 1 — the
off-diagonal structure appears only here. Cortical regions relate to each
other across time, not within a moment.*
```

Also add the architecture flow diagram you already have:

```bash
cp /path/to/transformer_flow.png docs/figures/
```

and put it at the top of the **Model** section.

**Which figures to use:** if the repository is public, the safe choice is
figures generated from the synthetic data, with a caption saying so. Real-data
figures go in once your PI has cleared them.

---

## Step 7 — Add the original notebook

```bash
jupyter nbconvert --clear-output --inplace your_notebook.ipynb
cp your_notebook.ipynb notebooks/ssl_spatiotemporal_transformer.ipynb
```

Clearing outputs first matters: cell outputs embed images as base64, which
makes the file large and every diff unreadable.

---

## Step 8 — Your profile

Create a repository named exactly your GitHub username. Its `README.md` renders
at the top of your profile page. Three or four lines — who you are, what you
work on, what you are looking for — is plenty.

Also worth doing once: [GitHub Student Developer Pack](https://education.github.com/pack)
with your Yale email.

---

## Afterwards

Things that are easy later and not worth blocking the first push on:

- **Wiki** for long tutorials, keeping the README scannable
- **Topics/tags** on the repository: `neuroscience`, `transformer`,
  `self-supervised-learning`, `calcium-imaging`, `alzheimers`
- **Zenodo** for the data, once it can be released, with the DOI in the README
- **Citation** updated from "Manuscript in preparation" to the real preprint

---

## A note on commits

You do not need a perfect history. Commit whenever something works:

```bash
git add -A
git commit -m "Add lag-split region-to-region rollout"
git push
```

The value of this repository on day one is that your code stops living only in
a Colab session that can disappear. Everything else is polish.
