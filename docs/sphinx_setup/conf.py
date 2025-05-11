# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import json
import shutil
from sphinx.util import logging
from json import JSONDecodeError
from sphinx.ext.autodoc import ClassDocumenter

project = 'OpenVINO™'
copyright = '2025, Intel®'
author = 'Intel®'

language = 'en'
version_name = 'nightly'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_inline_tabs',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'openvino_custom_sphinx_sitemap',
    'myst_parser',
    'breathe'
    ]

autodoc_mock_imports = []

try:
    import openvino
except ImportError:
    autodoc_mock_imports.append("openvino")
    autodoc_mock_imports.append("openvino_genai")  # Mock openvino_genai too, as it depends on openvino

if "openvino" not in autodoc_mock_imports:
    try:
        import openvino_genai
    except ImportError:
        autodoc_mock_imports.append("openvino_genai")


breathe_projects = {
    "openvino": "../xml/"
}

myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 4
suppress_warnings = [
    'misc.highlighting_failure',
    'myst.xref_missing'
    ]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}


# html_baseurl = 'https://docs.openvino.ai/2025/'


# -- Sitemap configuration ---------------------------------------------------

sitemap_url_scheme = "{link}"
site_url = f'https://docs.openvino.ai/{version_name}/'

ov_sitemap_urlset = [
    ("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9"),
    ("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance"),
    ("xmlns:coveo", "https://www.coveo.com/en/company/about-us"),
    ("xsi:schemaLocation", "http://www.sitemaps.org/schemas/sitemap/0.9 http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd")
]

ov_sitemap_meta = [
    ('coveo:metadata', {
        'ovversion': version_name,
        'ovdoctype': 'null',
        'ovcategory': 'null'
    })
]

# ----------------------------------------------------

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
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'openvino_sphinx_theme'

html_theme_path = ['_themes']


html_theme_options = {
    "navigation_depth": 8,
    "show_nav_level": 2,
    "use_edit_page_button": True,
    "github_url": "https://github.com/openvinotoolkit/openvino",
    # "footer_items": ["footer_info"],
    "show_prev_next": False,
}

html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"]
}

html_context = {
    'current_language': 'English',
    #'languages': (('English', '/latest'), ('Chinese', '/cn/latest')),
    'doxygen_mapping_file': '@DOXYGEN_MAPPING_FILE@',
    # go back fours 3 steps down in directory to reach openvino dir
    'doxygen_snippet_root': os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')),
    'default_mode': 'light'
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
    doxygen_mapping_file = os.path.dirname(os.getcwd()) + '/mapping.json'
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

html_css_files = [
    'css/openvino_sphinx_theme.css',
    'css/button.css',
    'css/input.css',
    'css/textfield.css',
    'css/tabs.css',
    'https://cdn.jsdelivr.net/npm/@splidejs/splide@4.1.4/dist/css/splide.min.css',
]

html_js_files = [
    'https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js',
    'js/openvino_sphinx_theme.js',
    'js/splide.min.js',
    'js/sortable_tables.js',
    'js/graphs.js',
    'js/gsearch.js',
    'js/hide_banner.js',
    'js/newsletter.js',
    'js/open_sidebar.js',
    'js/papaparse.min.js',
    'js/viewer.min.js',
    'js/custom.js',
    'js/modern.js',
]

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


def replace_design_tabs_script(app, exception):
    shutil.copy("../../../docs/sphinx_setup/_static/design-tabs.js","../_build/_static/design-tabs.js")


def setup(app):
    logger = logging.getLogger(__name__)
    app.add_config_value('doxygen_mapping_file',
                         doxygen_mapping_file, rebuild=True)
    app.add_config_value('repositories', repositories, rebuild=True)
    app.connect('autodoc-skip-member', autodoc_skip_member)
    # app.connect('build-finished', replace_design_tabs_script)
