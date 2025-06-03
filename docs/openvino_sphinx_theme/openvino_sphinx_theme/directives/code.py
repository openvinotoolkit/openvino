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
import html
import csv

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


class DataTable(Directive):
    required_arguments = 0
    has_content = False
    option_spec = {'header-rows': directives.nonnegative_int,
                   'file': directives.path,
                   'class': directives.unchanged,
                   'name': directives.unchanged,
                   'data-column-hidden': directives.unchanged,
                   'data-page-length': directives.unchanged,
                   'data-order': directives.unchanged
                   }

    def run(self) -> List[Node]:
        current_directory = os.path.dirname(os.path.abspath(self.state.document.current_source))
        csv_file = os.path.normpath(os.path.join(current_directory, self.options['file']))
        if os.path.isfile(csv_file) is False:
            self.warning("Cannot find the specified CSV file. "
                         "Please provide a correct path.")
        csv_node = []
        with open(csv_file, 'r') as j:
            csv_data = list(csv.reader(j))
            class_table_tag = f' class="{html.escape(self.options["class"])}"' if "class" in self.options else ""
            id_table_tag = f' id="{html.escape(self.options["name"])}"' if "name" in self.options else ""
            data_column_hidden_tag = f' data-column-hidden="{html.escape(self.options["data-column-hidden"])}"' if "data-column-hidden" in self.options else ""
            data_order_tag = f' data-order="{html.escape(self.options["data-order"])}"' if "data-order" in self.options else ""
            data_page_length_tag = f' data-page-length="{html.escape(self.options["data-page-length"])}"' if "data-page-length" in self.options else ""
            csv_table_html = f'<table{class_table_tag}{id_table_tag}{data_column_hidden_tag}{data_order_tag}{data_page_length_tag}>'
            head_rows = 0
            head_rows += self.options.get('header-rows', 0)
            row_count = 0
            for row in csv_data[:head_rows]:
                row_count += 1
                parity = "row-even" if row_count % 2 == 0 else "row-odd"
                csv_table_html += '<thead><tr class="' + parity + '">'
                for value in row:
                    csv_table_html += '<th class="head"><p>%s</p></th>' % value
                csv_table_html += '</tr></thead>\n'
            csv_table_html += '<tbody>'
            for row in csv_data[head_rows:]:
                row_count += 1
                parity = "row-even" if row_count % 2 == 0 else "row-odd"
                csv_table_html += '<tr class="' + parity + '">'
                for value in row:
                    csv_table_html += '<td><p>%s</p></td>' % value
            csv_table_html += '</tr>\n</tbody>'
            csv_table_html += "</tr>"
            csv_table_html += '</tbody></table>'
        csv_node.append(nodes.raw(csv_table_html, csv_table_html, format="html"))

        return csv_node