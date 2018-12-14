"""
 Copyright (c) 2018 Intel Corporation

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


class BasicError(Exception):
    """ Base class for all exceptions in Model Optimizer

        It operates like Exception but when it is converted to str,
        it formats string as args[0].format(*args[1:]), where
        args are arguments provided when an exception instance is
        created.
    """

    def __str__(self):
        if len(self.args) <= 1:
            return Exception.__str__(self)
        return self.args[0].format(*self.args[1:])  # pylint: disable=unsubscriptable-object


class FrameworkError(BasicError):
    """ User-friendly error: raised when the error on the framework side. """
    pass


class Error(BasicError):
    """ User-friendly error: raised when the error on the user side. """
    pass


class InternalError(BasicError):
    """ Not user-friendly error: user cannot fix it and it points to the bug inside MO. """
    pass

