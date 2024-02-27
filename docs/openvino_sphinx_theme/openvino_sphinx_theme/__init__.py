import os
import sys
import json
from json import JSONDecodeError
from sphinx.errors import ExtensionError
import jinja2
from docutils.parsers import rst
from pathlib import Path
from bs4 import BeautifulSoup
from sphinx.util import logging
from pydata_sphinx_theme import index_toctree
from .directives.code import DoxygenSnippet, Scrollbox, Nodescrollbox, visit_scrollbox, depart_scrollbox, Showcase, Nodeshowcase, visit_showcase, depart_showcase

SPHINX_LOGGER = logging.getLogger(__name__)


def setup_edit_url(app, pagename, templatename, context, doctree):
    """Add a function that jinja can access for returning the edit URL of a page."""

    def has_github_page():
        doxygen_mapping_file = app.config.html_context.get('doxygen_mapping_file')
        name = pagename.rsplit('-')[0]
        if name in doxygen_mapping_file:
            return True
        return False

    def get_edit_url():
        """Return a URL for an "edit this page" link."""
        doc_context = dict()
        doc_context.update(**context)

        # Make sure that doc_path has a path separator only if it exists (to avoid //)
        doc_path = doc_context.get("doc_path", "")
        if doc_path and not doc_path.endswith("/"):
            doc_path = f"{doc_path}/"

        # ensure custom URL is checked first, if given
        url_template = doc_context.get("edit_page_url_template")

        if url_template is not None:
            if "file_name" not in url_template:
                raise ExtensionError(
                    "Missing required value for `use_edit_page_button`. "
                    "Ensure `file_name` appears in `edit_page_url_template`: "
                    f"{url_template}"
                )
            return jinja2.Template(url_template).render(**doc_context)

        url_template = '{{ github_url }}/{{ github_user }}/{{ github_repo }}' \
                       '/edit/{{ github_version }}/{{ doc_path }}{{ file_name }}'

        doxygen_mapping_file = app.config.html_context.get('doxygen_mapping_file')
        rst_name = pagename.rsplit('-')[0]
        file_name = doxygen_mapping_file[rst_name]
        parent_folder = Path(os.path.dirname(file_name)).parts[0]
        file_name = Path(*Path(file_name).parts[1:]).as_posix()

        doc_context.update(doc_path=doc_path, file_name=file_name)
        try:
            repositories = app.config.repositories
        except AttributeError:
            raise ExtensionError("Missing required value for `use_edit_page_button`. "
                                 "Ensure `repositories` is set in conf.py.")

        required = ['github_user', 'github_repo', 'github_version', 'host_url']
        for repo, config in repositories.items():
            for key, val in config.items():
                if key not in required or not val:
                    raise ExtensionError(f'Missing required value for `{repo}` entry in `repositories`'
                                         f'Ensure {required} all set.')
            if parent_folder == repo:
                doc_context.update(github_user=config['github_user'])
                doc_context.update(github_repo=config['github_repo'])
                doc_context.update(github_version=config['github_version'])
                doc_context.update(github_url=config['host_url'])
                return jinja2.Template(url_template).render(**doc_context)

    context["get_edit_url"] = get_edit_url
    context['has_github_page'] = has_github_page()

    # Ensure that the max TOC level is an integer
    context["theme_show_toc_level"] = int(context.get("theme_show_toc_level", 1))


def get_theme_path():
    theme_path = os.path.abspath(os.path.dirname(__file__))
    return theme_path


def add_toctree_functions(app, pagename, templatename, context, doctree):

    # override pydata_sphinx_theme
    def generate_sidebar_nav(kind, startdepth=None, **kwargs):
        """
        Return the navigation link structure in HTML. Arguments are passed
        to Sphinx "toctree" function (context["toctree"] below).

        We use beautifulsoup to add the right CSS classes / structure for bootstrap.

        See https://www.sphinx-doc.org/en/master/templating.html#toctree.

        Parameters
        ----------
        kind : ["navbar", "sidebar", "raw"]
            The kind of UI element this toctree is generated for.
        startdepth : int
            The level of the toctree at which to start. By default, for
            the navbar uses the normal toctree (`startdepth=0`), and for
            the sidebar starts from the second level (`startdepth=1`).
        kwargs: passed to the Sphinx `toctree` template function.

        Returns
        -------
        HTML string (if kind in ["navbar", "sidebar"])
        or BeautifulSoup object (if kind == "raw")
        """

        open_first = False
        if 'open_first' in kwargs:
            open_first = kwargs.pop('open_first')

        if startdepth is None:
            startdepth = 1 if kind == "sidebar" else 0

        if startdepth == 0:
            toc_sphinx = context["toctree"](**kwargs)
        else:
            # select the "active" subset of the navigation tree for the sidebar
            toc_sphinx = index_toctree(app, pagename, startdepth, **kwargs)

        soup = BeautifulSoup(toc_sphinx, "html.parser")

        if kind == "sidebar":
            # Add bootstrap classes for first `ul` items
            for ul in soup("ul", recursive=False):
                ul.attrs["class"] = ul.attrs.get("class", []) + ["nav", "bd-sidenav"]

        toctree_checkbox_count = 0

        for li in soup.find_all("li"):
            # pair "current" with "active" since that's what we use w/ bootstrap
            if "current" in li["class"]:
                li["class"].append("active")

            # Remove navbar/sidebar links to sub-headers on the page
            if li.find("a"):
                href = li.find("a")["href"]
                if "#" in href and href != "#":
                    li.decompose()
                    continue

            if kind == "navbar":
                li["class"].append("nav-item")
                li.find("a")["class"].append("nav-link")
                # only select li items (not eg captions)
                # out = "\n".join([ii.prettify() for ii in soup.find_all("li")])
            elif kind == "sidebar":
                if li is None:
                    continue
                # We check all "li" elements, to add a "current-page" to the correct li.
                classes = li.get("class", [])

                # Nothing more to do, unless this has "children"
                if not li.find("ul"):
                    continue

                # Add a class to indicate that this has children.
                li["class"] = classes + ["has-children"]

                # We're gonna add a checkbox.
                toctree_checkbox_count += 1
                checkbox_name = f"toctree-checkbox-{toctree_checkbox_count}"

                # Add the "label" for the checkbox which will get filled.
                if soup.new_tag is None:
                    continue
                label = soup.new_tag("label", attrs={"for": checkbox_name})
                label.append(soup.new_tag("i", attrs={"class": "fas fa-chevron-down"}))
                li.insert(1, label)

                # Add the checkbox that's used to store expanded/collapsed state.
                checkbox = soup.new_tag(
                    "input",
                    attrs={
                        "type": "checkbox",
                        "class": ["toctree-checkbox"],
                        "id": checkbox_name,
                        "name": checkbox_name,
                    },
                )
                # if this has a "current" class, be expanded by default
                # (by checking the checkbox)
                if "current" in classes or (open_first and toctree_checkbox_count == 1):
                    checkbox.attrs["checked"] = ""
                li.insert(1, checkbox)

        return soup

    context["generate_sidebar_nav"] = generate_sidebar_nav


def read_doxygen_configs(app, env, docnames):
    if app.config.html_context.get('doxygen_mapping_file'):
        try:
            with open(app.config.html_context.get('doxygen_mapping_file'), 'r', encoding='utf-8') as f:
                app.config.html_context['doxygen_mapping_file'] = json.load(f)
        except (JSONDecodeError, FileNotFoundError):
            app.config.html_context['doxygen_mapping_file'] = dict()


def setup(app):
    theme_path = get_theme_path()
    templates_path = os.path.join(theme_path, 'templates')
    static_path = os.path.join(theme_path, 'static')
    app.config.templates_path.append(templates_path)
    app.config.html_static_path.append(static_path)
    app.connect("html-page-context", setup_edit_url, priority=sys.maxsize)
    app.connect("html-page-context", add_toctree_functions)
    app.connect('env-before-read-docs', read_doxygen_configs)
    app.add_html_theme('openvino_sphinx_theme', theme_path)
    rst.directives.register_directive('doxygensnippet', DoxygenSnippet)
    rst.directives.register_directive('scrollbox', Scrollbox)
    rst.directives.register_directive('showcase', Showcase)
    app.add_node(
        Nodescrollbox,
        html=(visit_scrollbox, depart_scrollbox),
        latex=(visit_scrollbox, depart_scrollbox)
    )
    app.add_node(
    Nodeshowcase,
    html=(visit_showcase, depart_showcase),
    latex=(visit_showcase, depart_showcase)
    )
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
