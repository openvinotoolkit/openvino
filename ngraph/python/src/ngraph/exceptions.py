# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ngraph exceptions hierarchy. All exceptions are descendants of NgraphError."""

## @defgroup ngraph_python_exceptions Exceptions
# ngraph exceptions hierarchy. All exceptions are descendants of NgraphError.
# @ingroup ngraph_python_api


## @ingroup ngraph_python_exceptions
class NgraphError(Exception):
    """Base class for Ngraph exceptions."""


## @ingroup ngraph_python_exceptions
class UserInputError(NgraphError):
    """User provided unexpected input."""


## @ingroup ngraph_python_exceptions
class NgraphTypeError(NgraphError, TypeError):
    """Type mismatch error."""
