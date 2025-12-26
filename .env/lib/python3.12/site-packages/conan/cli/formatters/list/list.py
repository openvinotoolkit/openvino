import json
import os

from jinja2 import Template, select_autoescape

from conan.api.output import cli_out_write
from conan.cli.formatters.list.search_table_html import list_packages_html_template
from conan import __version__


def list_packages_html(result):
    results = result["results"]
    cli_args = result["cli_args"]
    conan_api = result["conan_api"]
    template_folder = os.path.join(conan_api.cache_folder, "templates")
    user_template = os.path.join(template_folder, "list_packages.html")
    template = list_packages_html_template
    if os.path.isfile(user_template):
        with open(user_template, 'r', encoding="utf-8", newline="") as handle:
            template = handle.read()
    template = Template(template, autoescape=select_autoescape(['html', 'xml']))
    content = template.render(results=json.dumps(results), base_template_path=template_folder,
                              version=__version__, cli_args=cli_args)
    cli_out_write(content)
