"""
 Copyright (C) 2018-2020 Intel Corporation

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
from extensions.load.loader import LoadFinish
from mo.graph.graph import Graph
from mo.utils import class_registration
from mo.utils.error import Error


class AnalysisResults:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AnalysisResults, cls).__new__(cls, *args, **kwargs)
            cls.results = {}
            cls.messages = []
        return cls._instance

    def __getattr__(self, item):
        return self.results[item]

    def __setattr__(self, key, value):
        self.results[key] = value

    @classmethod
    def get_result(cls, key=None):
        if key is not None:
            if key in cls.results and cls.results[key] is not None:
                return cls.results[key]
        else:
            return cls.results

    @classmethod
    def add_result(cls, result, key=None):
        if key is not None:
            cls.results[key] = result
        else:
            cls.results.update(result)

    @classmethod
    def get_messages(cls):
        return cls.messages

    @classmethod
    def add_message(cls, message):
        cls.messages.append(message)


class AnalyzeAction(object):
    registered_cls = []
    registered_ops = {}
    excluded_replacers = []
    run_not_recursively = True

    def find_and_replace_pattern(self, graph: Graph):
        analysis_results = AnalysisResults()
        failed_analysers = []

        try:
            result, msg = self.analyze(graph)  # pylint: disable=assignment-from-no-return
        except SystemExit:
            # the analysis transformation printing analysis results to the screen calls sys.exit(0) which in fact raises
            # SystemExit exception, so we handle it here
            sys.exit(0)
        except:
            failed_analysers.append(str(self.__class__))
            analysis_results.add_result(failed_analysers, 'failed_analysers')
            result = None
            msg = None

        if result is not None:
            analysis_results.add_result(result)
        if msg is not None:
            analysis_results.add_message(msg)

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
        return [LoadFinish]

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
