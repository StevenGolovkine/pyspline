# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "pyspline"
copyright = "2024, Steven Golovkine"
author = "Steven Golovkine"
release = "v0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_gallery.gen_gallery",
]

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_permalinks_icon = "<span>#</span>"
html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]


sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": "../examples",
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
    "reference_url": {
        # The module you locally document uses None
        "pyspline": None,
    },
    "backreferences_dir": "backreferences",
    "doc_module": "pyspline",
}

autosummary_generate = True
autodoc_member_order = "bysource"

autodoc_default_options = {"show-inheritance": True}
