import os.path

from sphinx.directives.code import LiteralInclude, LiteralIncludeReader, container_wrapper
from sphinx.util import logging
from docutils.parsers.rst import directives
from typing import List, Tuple
from docutils.nodes import Node
from docutils import nodes
from sphinx.util import parselinenos

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
