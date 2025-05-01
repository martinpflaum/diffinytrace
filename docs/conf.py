# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
#%%
import os

try:
    # Path to the index.rst file
    index_rst_path = r"index.rst"

    # Directory to save the generated .rst files
    docs_dir = os.path.dirname(index_rst_path)

    # Read the content of index.rst
    with open(index_rst_path, "r") as file:
        lines = file.readlines()

    # Find the "Submodules" section and extract submodule names
    submodules_start = None
    submodules = []
    for i, line in enumerate(lines):
        if line.strip() == "Submodules":
            submodules_start = i
        elif submodules_start is not None and line.startswith(".. automodule::"):
            print("line:",line)
            submodule_name = line.split("::")[1].strip()
            submodules.append(submodule_name)
        elif submodules_start is not None and line.strip() == "":
            pass
    # Generate .rst files for each submodule
    for submodule in submodules:
        rst_file_path = os.path.join(docs_dir, f"{submodule.split('.')[-1]}.rst")
        with open(rst_file_path, "w") as rst_file:
            rst_file.write(f"{submodule.split('.')[-1]}\n")
            rst_file.write("=" * (len(submodule) + 7) + "\n\n")
            rst_file.write(f".. automodule:: {submodule}\n")
            rst_file.write("   :members:\n")
            rst_file.write("   :undoc-members:\n")
            rst_file.write("   :show-inheritance:\n")

    # Remove everything after "Submodules" in index.rst
    with open(index_rst_path, "w") as file:
        for line in lines[:submodules_start-1]:  # Keep lines up to "Submodules"
            if line != "Submodules\n" or "":
                file.write(line)

        # Add the generated .rst files to the first toctree
        #file.write("\n.. toctree::\n   :maxdepth: 4\n\n")
        for submodule in submodules:
            file.write(f"   {submodule.split('.')[-1]}\n")
except:
    pass
#%%

project = 'diffinytrace'
copyright = '2025, Martin Pflaum'
author = 'Martin Pflaum'
release = '2.1'

import os
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#sys.path.insert(0, os.path.abspath("."))
#sys.path.insert(0, os.path.abspath("../../diffinytrace"))
sys.path.insert(0, os.path.abspath(".."))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc","sphinx.ext.autosummary","sphinx.ext.napoleon","sphinx.ext.viewcode"]

"""
extensions = [
#    "sphinx.ext.apidoc",             # Generate API documentation from Python packages
 #   "sphinx.ext.autodoc",            # Include documentation from docstrings
#    "sphinx.ext.autosectionlabel",   # Allow referencing sections by their title
    "sphinx.ext.autosummary",        # Generate autodoc summaries
#    "sphinx.ext.coverage",           # Collect doc coverage stats
    #"sphinx.ext.doctest",            # Test snippets in the documentation
    #"sphinx.ext.duration",           # Measure durations of Sphinx processing
    #"sphinx.ext.extlinks",           # Markup to shorten external links
    #"sphinx.ext.githubpages",        # Publish HTML docs in GitHub Pages
    #"sphinx.ext.graphviz",           # Add Graphviz graphs
    #"sphinx.ext.ifconfig",           # Include content based on configuration
    #"sphinx.ext.imgconverter",       # A reference image converter using ImageMagick
    #"sphinx.ext.inheritance_diagram",# Include inheritance diagrams
    "sphinx.ext.intersphinx",        # Link to other projects’ documentation
#    "sphinx.ext.linkcode",           # Add external links to source code
    "sphinx.ext.mathjax",            # Math support for HTML outputs in Sphinx
    "sphinx.ext.napoleon",           # Support for NumPy and Google style docstrings
    #"sphinx.ext.todo",               # Support for todo items
    "sphinx.ext.viewcode",           # Add links to highlighted source code
    "sphinx_rtd_theme"]"""

templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme = "furo"
html_static_path = ['_static']
