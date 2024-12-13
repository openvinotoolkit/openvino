# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generic utilities. Factor related functions out to separate files."""

from openvino._pyopenvino.util import numpy_to_c, replace_node, replace_output_update_name

from openvino.utils.package_utils import _add_openvino_libs_to_search_path
from openvino.utils.package_utils import get_cmake_path
from openvino.utils.package_utils import deprecated
from openvino.utils.package_utils import _ClassPropertyDescriptor
from openvino.utils.package_utils import classproperty
from openvino.utils.package_utils import deprecatedclassproperty
