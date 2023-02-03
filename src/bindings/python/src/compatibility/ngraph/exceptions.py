# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ngraph exceptions hierarchy. All exceptions are descendants of NgraphError."""


class NgraphError(Exception):
    """Base class for Ngraph exceptions."""


class UserInputError(NgraphError):
    """User provided unexpected input."""


class NgraphTypeError(NgraphError, TypeError):
    """Type mismatch error."""
