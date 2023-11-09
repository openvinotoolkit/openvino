import os.path
from pathlib import Path
import sys
from sphinx.directives.code import LiteralInclude, LiteralIncludeReader, container_wrapper
from sphinx.util import logging
from docutils.parsers.rst import Directive, directives
from typing import List, Tuple
from docutils.nodes import Node
from docutils import nodes
from sphinx.util import parselinenos
import requests
import re
import json

logger = logging.getLogger(__name__)


class DoxygenSnippet(LiteralInclude):

    option_spec = dict({'fragment': directives.unchanged_required}, **LiteralInclude.option_spec)

    def run(self) -> List[Node]:
        if 'fragment' in self.options:
            self.options['start-after'] = self.options['fragment']
            self.options['end-before'] = self.options['fragment']
        document = self.state.document
        if not document.settings.file_insertion_enabled:
            return [document.reporter.warning('File insertion disabled',
                                              line=self.lineno)]
        # convert options['diff'] to absolute path
        if 'diff' in self.options:
            _, path = self.env.relfn2path(self.options['diff'])
            self.options['diff'] = path

        try:
            location = self.state_machine.get_source_and_line(self.lineno)
            doxygen_snippet_root = self.config.html_context.get('doxygen_snippet_root')

            if doxygen_snippet_root and os.path.exists(doxygen_snippet_root):
                rel_filename = self.arguments[0]
                filename = os.path.join(doxygen_snippet_root, rel_filename)
            else:
                rel_filename, filename = self.env.relfn2path(self.arguments[0])
            self.env.note_dependency(rel_filename)

            reader = LiteralIncludeReader(filename, self.options, self.config)
            text, lines = reader.read(location=location)

            retnode = nodes.literal_block(text, text, source=filename)  # type: Element
            retnode['force'] = 'force' in self.options
            self.set_source_info(retnode)
            if self.options.get('diff'):  # if diff is set, set udiff
                retnode['language'] = 'udiff'
            elif 'language' in self.options:
                retnode['language'] = self.options['language']
            if ('linenos' in self.options or 'lineno-start' in self.options or
                    'lineno-match' in self.options):
                retnode['linenos'] = True
            retnode['classes'] += self.options.get('class', [])
            extra_args = retnode['highlight_args'] = {}
            if 'emphasize-lines' in self.options:
                hl_lines = parselinenos(self.options['emphasize-lines'], lines)
                if any(i >= lines for i in hl_lines):
                    logger.warning(__('line number spec is out of range(1-%d): %r') %
                                   (lines, self.options['emphasize-lines']),
                                   location=location)
                extra_args['hl_lines'] = [x + 1 for x in hl_lines if x < lines]
            extra_args['linenostart'] = reader.lineno_start

            if 'caption' in self.options:
                caption = self.options['caption'] or self.arguments[0]
                retnode = container_wrapper(self, retnode, caption)

            # retnode will be note_implicit_target that is linked from caption and numref.
            # when options['name'] is provided, it should be primary ID.
            self.add_name(retnode)

            return [retnode]
        except Exception as exc:
            return [document.reporter.warning(exc, line=self.lineno)]


def visit_scrollbox(self, node):
    attrs = {}
    attrs["style"] = (
        (("height:" + "".join(c for c in str(node["height"]) if c.isdigit()) + "px!important; " ) if "height" in node is not None else "")
        + (("width:" + "".join(c for c in str(node["width"]) if c.isdigit()) ) if "width" in node is not None else "")
        + (("px; " if node["width"].find("px") != -1 else "%;") if "width" in node is not None else "")
        + ( ("border-left:solid "+"".join(c for c in str(node["delimiter"]) if c.isdigit())+ "px " + (("".join(str(node["delimiter-color"]))) if "delimiter-color" in node is not None else "#dee2e6") +"; ") if "delimiter" in node is not None else "")
    )
    attrs["class"] = "scrollbox"
    self.body.append(self.starttag(node, "div", **attrs))


def depart_scrollbox(self, node):
    self.body.append("</div>\n")


class Nodescrollbox(nodes.container):
    def create_scrollbox_component(
        rawtext: str = "",
        **attributes,
    ) -> nodes.container:
        node = nodes.container(rawtext, is_div=True, **attributes)
        return node


class Scrollbox(Directive):
    has_content = True
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {
        'name': directives.unchanged,
        'width': directives.length_or_percentage_or_unitless,
        'height': directives.length_or_percentage_or_unitless,
        'style': directives.unchanged,
        'delimiter': directives.length_or_percentage_or_unitless,
        'delimiter-color': directives.unchanged,
    }

    has_content = True

    def run(self):
        classes = ['scrollbox','']
        node = Nodescrollbox("div", rawtext="\n".join(self.content), classes=classes)
        if 'height' in self.options:
            node['height'] = self.options['height']
        if 'width' in self.options:
            node['width'] = self.options['width']
        if 'delimiter' in self.options:
            node['delimiter'] = self.options['delimiter']
        if 'delimiter-color' in self.options:
            node['delimiter-color'] = self.options['delimiter-color']
        self.add_name(node)
        if self.content:
            self.state.nested_parse(self.content, self.content_offset, node)
        return [node]

def visit_showcase(self, node):
    attrs = {}
    notebook_file = ("notebooks/" + node["title"] + "-with-output.html") if 'title' in node is not None else ""
    link_title = (node["title"]) if 'title' in node is not None else "OpenVINO Interactive Tutorial"

    if "height" or "width" in node:
        attrs["style"] = (
            (("height:" + "".join(c for c in str(node["height"]) if c.isdigit()) + "px!important; " ) if "height" in node is not None else "")
            + (("width:" + "".join(c for c in str(node["width"]) if c.isdigit()) ) if "width" in node is not None else "")
            + (("px; " if node["width"].find("px") != -1 else "%;") if "width" in node is not None else "")
        )
    self.body.append("<div class='showcase-wrap'>")
    self.body.append(self.starttag(node, "div", **attrs))
    self.body.append(("<div class='showcase-img-placeholder'><a href='" + notebook_file + "' title='" + link_title + "'><img " + (" class='" + (node["img-class"] + " showcase-img' ") if 'img-class' in node is not None else " class='showcase-img'") + "src='" + node["img"] + "' alt='"+os.path.basename(node["img"])+"' /></a></div>") if "img" in node is not None else "")
    self.body.append("<div class='showcase-content'><div class='showcase-content-container'>")


def depart_showcase(self, node):
    notebooks_repo = "https://github.com/openvinotoolkit/openvino_notebooks/blob/main/"
    notebooks_binder = "https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath="
    notebooks_colab = "https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/"
    git_badge = "<img class='showcase-badge' src='https://badgen.net/badge/icon/github?icon=github&amp;label' alt='Github'>"
    binder_badge = "<img class='showcase-badge' src='https://mybinder.org/badge_logo.svg' alt='Binder'>"
    colab_badge = "<img class='showcase-badge' src='https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667' alt='Colab'>"
    binder_list_file = Path('../../../docs/notebooks/notebooks_with_binder_buttons.txt').resolve(strict=True)
    colab_list_file = Path('../../../docs/notebooks/notebooks_with_colab_buttons.txt').resolve(strict=True)
    openvino_notebooks_repo_listing = Path('../../../docs/notebooks/openvino_notebooks.json').resolve(strict=True)
    with open(binder_list_file, 'r+', encoding='cp437') as file:
        binder_buttons_list = file.read().splitlines()
    with open(colab_list_file, 'r+', encoding='cp437') as file:
        colab_buttons_list = file.read().splitlines()
    if not os.path.exists(openvino_notebooks_repo_listing):
        print("No such JSON file", openvino_notebooks_repo_listing)
    else:
        result = open(openvino_notebooks_repo_listing, 'r').read()

    paths_list = json.loads(result)

    if "tree" in paths_list:
        list_all_paths = [p.get('path') for p in paths_list['tree'] if p.get('path')]
        ipynb_list = [x for x in list_all_paths if re.match("notebooks/[0-9]{3}.*\.ipynb$", x)]
        notebook_with_ext = node["title"] + ".ipynb"
        matched_notebook = [match for match in ipynb_list if notebook_with_ext in match]
    else:
        raise Exception('Key "tree" is not present in the JSON file')

    notebook_file = ("notebooks/" + node["title"] + "-with-output.html") if 'title' in node is not None else ""
    link_title = (node["title"]) if 'title' in node is not None else "OpenVINO Interactive Tutorial"

    self.body.append(("<a href='" + notebook_file + "' title='" + link_title + "'><h2 class='showcase-title'>" + node["title"] + "</h2></a>") if 'title' in node is not None else "")

    if matched_notebook is not None:
        for n in matched_notebook:
            self.body.append(("<a href='" + notebooks_repo + n + "' target='_blank'>" + git_badge + "</a>") if 'title' in node is not None else "")
            if node["title"] in binder_buttons_list:
                self.body.append(("<a href='" + notebooks_binder + n + "' target='_blank'>" + binder_badge + "</a>"
                                  ) if 'title' in node is not None else "")
            if node["title"] in colab_buttons_list:
                self.body.append(("<a href='" + notebooks_colab + n + "' target='_blank'>" + colab_badge + "</a>"
                                  ) if 'title' in node is not None else "")

    self.body.append("</div><button class='showcase-button' type='button' title='" + link_title +
                     "' onclick=\"location.href='" + notebook_file + "'\">Read more</a></div></div></div>\n")


class Nodeshowcase(nodes.container):
    def create_showcase_component(
        rawtext: str = "",
        **attributes,
    ) -> nodes.container:
        node = nodes.container(rawtext, is_div=True, **attributes)
        return node


class Showcase(Directive):
    has_content = True
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {
        'class': directives.class_option,
        'name': directives.unchanged,
        'width': directives.length_or_percentage_or_unitless,
        'height': directives.length_or_percentage_or_unitless,
        'style': directives.unchanged,
        'img': directives.unchanged,
        'img-class': directives.unchanged,
        'title': directives.unchanged,
        'git': directives.unchanged,
    }

    has_content = True

    def run(self):

        classes = ['showcase']
        node = Nodeshowcase("div", rawtext="\n".join(self.content), classes=classes)
        if 'height' in self.options:
            node['height'] = self.options['height']
        if 'width' in self.options:
            node['width'] = self.options['width']
        if 'img' in self.options:
            node['img'] = self.options['img']
        if 'img-class' in self.options:
            node['img-class'] = self.options['img-class']
        if 'title' in self.options:
            node['title'] = self.options['title']
        if 'git' in self.options:
            node['git'] = self.options['git']
        node['classes'] += self.options.get('class', [])
        self.add_name(node)
        if self.content:
            self.state.nested_parse(self.content, self.content_offset, node)
        return [node]
