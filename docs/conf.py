# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
#%%
import os
import subprocess

# Step 1: Run sphinx-apidoc to generate .rst files
docs_dir = os.path.abspath(".")
source_dir = os.path.abspath("../diffinytrace")  # Path to the source directory
cmd = ['sphinx-apidoc']
cmd.append('-f')
# Add other necessary arguments to the command
cmd.append('-o')
cmd.append(docs_dir)  # Output directory
cmd.append(source_dir)  # Source directory

try:
    res = subprocess.run(cmd, text=True, capture_output=True, check=True)
    print(res)
    print(res.stdout)

except subprocess.CalledProcessError as e:
    print("CalledProcessError: " + str(e))
    print("Error Output: " + e.output)



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
