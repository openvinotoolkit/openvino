# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import json
import shutil
from sphinx.util import logging
from json import JSONDecodeError
from sphinx.ext.autodoc import ClassDocumenter


sys.path.insert(0, os.path.abspath('doxyrest-sphinx'))

# -- Project information -----------------------------------------------------

project = 'OpenVINO™'
copyright = '2022, Intel®'
author = 'Intel®'

language = 'en'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_inline_tabs',
    'sphinx_copybutton',
    'sphinx_panels',
    'sphinx_design',
    'doxyrest',
    'cpplexer',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_sitemap'
]

html_baseurl = 'https://docs.openvino.ai/latest/'

sitemap_url_scheme = "{link}"

html_favicon = '_static/favicon.ico'
autodoc_default_flags = ['members']
autosummary_generate = True
autosummary_imported_members = True

html_logo = '_static/logo.svg'
html_copy_source = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


panels_add_bootstrap_css = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "openvino_sphinx_theme"

html_theme_path = ['_themes']

html_theme_options = {
    "navigation_depth": 8,
    "show_nav_level": 2,
    "use_edit_page_button": True,
    "github_url": "https://github.com/openvinotoolkit/openvino",
    "footer_items": ["footer_info"],
    "show_prev_next": False,
}

html_context = {
    'current_language': 'English',
    'languages': (('English', '/latest'), ('Chinese', '/cn/latest')),
    'doxygen_mapping_file': '@DOXYGEN_MAPPING_FILE@',
    'doxygen_snippet_root': '@OpenVINO_SOURCE_DIR@'
}

repositories = {
    'openvino': {
        'github_user': 'openvinotoolkit',
        'github_repo': 'openvino',
        'github_version': 'master',
        'host_url': 'https://github.com'
    },
    'pot': {
        'github_user': 'openvinotoolkit',
        'github_repo': 'openvino',
        'github_version': 'master',
        'host_url': 'https://github.com'
    },
    'ote': {
        'github_user': 'openvinotoolkit',
        'github_repo': 'training_extensions',
        'github_version': 'develop',
        'host_url': 'https://github.com'
    },
    'open_model_zoo': {
        'github_user': 'openvinotoolkit',
        'github_repo': 'open_model_zoo',
        'github_version': 'master',
        'host_url': 'https://github.com'
    },
    'ovms': {
        'github_user': 'openvinotoolkit',
        'github_repo': 'model_server',
        'github_version': 'main',
        'host_url': 'https://github.com'
    }
}

try:
    doxygen_mapping_file = '@DOXYGEN_MAPPING_FILE@'
    with open(doxygen_mapping_file, 'r', encoding='utf-8') as f:
        doxygen_mapping_file = json.load(f)
except JSONDecodeError:
    doxygen_mapping_file = dict()
except FileNotFoundError:
    doxygen_mapping_file = dict()

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# monkeypatch sphinx api doc to prevent showing inheritance from object and enum.Enum
add_line = ClassDocumenter.add_line


def add_line_no_base_object(self, line, *args, **kwargs):
    if line.strip() in ['Bases: :class:`object`', 'Bases: :class:`enum.Enum`']:
        return
    else:
        add_line(self, line, *args, **kwargs)


ClassDocumenter.add_line = add_line_no_base_object

# OpenVINO Python API Reference Configuration
exclude_pyapi_methods = ('__weakref__',
                         '__doc__',
                         '__module__',
                         '__dict__',
                         'add_openvino_libs_to_path'
                         )


def autodoc_skip_member(app, what, name, obj, skip, options):
    return name in exclude_pyapi_methods


shutil.copy("../../../docs/home.rst",".")

def replace_index_with_redirect(app,exception):
    shutil.copy("../../../docs/index.html","../_build/index.html")

def setup(app):
    logger = logging.getLogger(__name__)
    app.add_config_value('doxygen_mapping_file',
                         doxygen_mapping_file, rebuild=True)
    app.add_config_value('repositories', repositories, rebuild=True)
    app.connect('autodoc-skip-member', autodoc_skip_member)
    app.connect('build-finished',replace_index_with_redirect)
    app.add_js_file('js/custom.js')
    app.add_js_file('js/graphs.js')
    app.add_js_file('js/graphs_ov_tf.js')
    app.add_js_file('js/open_sidebar.js')
    