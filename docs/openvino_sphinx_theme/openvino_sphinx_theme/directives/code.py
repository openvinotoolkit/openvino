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
        return []
