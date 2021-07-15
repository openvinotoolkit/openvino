from sphinx.errors import ExtensionError
import jinja2
import os
import json
from json import JSONDecodeError
import sys
from pathlib import Path
from sphinx.util import logging

SPHINX_LOGGER = logging.getLogger(__name__)


def setup_edit_url(app, pagename, templatename, context, doctree):
    """Add a function that jinja can access for returning the edit URL of a page."""

    def has_github_page():
        doxygen_mapping_file = app.config['doxygen_mapping_file']
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

        doxygen_mapping_file = app.config['doxygen_mapping_file']
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


def read_doxygen_mapping(app, config):
    doxygen_mapping_file = config['doxygen_mapping_file']
    try:
        with open(doxygen_mapping_file, 'r') as f:
            config['doxygen_mapping_file'] = json.load(f)
    except FileNotFoundError:
        ExtensionError('{}: file not found.'.format(doxygen_mapping_file))
    except JSONDecodeError as e:
        ExtensionError('{}: must be a json file.'.format(doxygen_mapping_file))


def get_theme_path():
    theme_path = os.path.abspath(os.path.dirname(__file__))
    return theme_path


def setup(app):
    theme_path = get_theme_path()
    app.connect('config-inited', read_doxygen_mapping)
    app.connect("html-page-context", setup_edit_url, priority=sys.maxsize)
    app.add_config_value('repositories', dict(), rebuild=True)
    app.add_config_value('doxygen_mapping_file', dict(), rebuild=True)
    app.add_html_theme('openvino_sphinx_theme', theme_path)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
