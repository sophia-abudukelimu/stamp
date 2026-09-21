# Notebooks

Put the original Colab notebook here, for reference:

    notebooks/ssl_spatiotemporal_transformer.ipynb

It is kept because it is the version the results were first produced with, and
because it is easier to read top to bottom than the package. It is **not** the
maintained code path — `train.py` and `analyze.py` are. If the two ever
disagree, the package is correct.

Before committing a notebook, clear its outputs:

    jupyter nbconvert --clear-output --inplace notebooks/*.ipynb

Cell outputs embed images as base64 and make diffs unreadable.
