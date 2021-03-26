# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ngraph.dll directory path visibility is needed to use _pyngraph module
# import below causes adding this path to os.environ["PATH"]
import ngraph  # noqa: F401 'imported but unused'
