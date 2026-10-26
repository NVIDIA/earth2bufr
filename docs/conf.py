# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Sphinx configuration for earth2bufr documentation."""

project = "earth2bufr"
copyright = "2025, NVIDIA CORPORATION & AFFILIATES"  # noqa: A001
author = "NVIDIA"
release = "0.1.1rc1"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
]

# AutoAPI configuration
autoapi_dirs = ["../src/earth2bufr"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_ignore = ["*/_lib*"]
autoapi_add_toctree_entry = False

# MyST parser
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Napoleon settings (numpydoc style)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
}

# Theme
html_theme = "furo"
html_title = "earth2bufr"
html_theme_options = {
    "source_repository": "https://github.com/NVIDIA/earth2bufr",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "superpowers"]
