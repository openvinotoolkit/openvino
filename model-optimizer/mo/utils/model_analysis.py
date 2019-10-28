"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import sys

from extensions.front.user_data_repack import UserDataRepack
from mo.graph.graph import Graph
from mo.utils import class_registration
from mo.utils.error import Error


class AnalyzeAction(object):
    registered_cls = []
    registered_ops = {}
    excluded_replacers = []
    run_not_recursively = True

    def find_and_replace_pattern(self, graph: Graph):
        if 'analysis_results' not in graph.graph:
            graph.graph['analysis_results'] = {'failed_analysers': []}

        try:
            result = self.analyze(graph)  # pylint: disable=assignment-from-no-return
        except SystemExit:
            # the analysis transformation printing analysis results to the screen calls sys.exit(0) which in fact raises
            # SystemExit exception, so we handle it here
            sys.exit(0)
        except:
            graph.graph['analysis_results']['failed_analysers'].append(str(self.__class__))
            result = None

        if result is not None:
            graph.graph['analysis_results'].update(result)

    def analyze(self, graph: Graph):
        raise Error('The method must be implemented in the sub-class')

    def run_before(self):
        """
        Returns list of replacer classes which this replacer must be run before.
        :return: list of classes
        """
        return [AnalysisCollectorAnchor, UserDataRepack]

    def run_after(self):
        """
        Returns list of replacer classes which this replacer must be run after.
        :return: list of classes
        """
        return []

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.FRONT_REPLACER


class AnalysisCollectorAnchor(AnalyzeAction):
    """
    All analyzers should depend on this one which is an anchor analyzer to develop custom post-processor of all
    analyzers results.
    """

    def run_before(self):
        return []

    def analyze(self, graph: Graph):
        pass


def graph_contains_scope(graph: Graph, scope: str):
    """
    Checks whether the graph contains node(s) which name starts with "scope" string.
    :param graph: graph to check
    :param scope: string defining the scope
    :return: the result of the check (True/False)
    """
    if scope[-1] != '/':
        scope += '/'
    return any([node.soft_get('name').startswith(scope) for node in graph.get_op_nodes()])
