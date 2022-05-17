#===============================================================================
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

include("cmake/Doxygen.cmake")
include("cmake/Doxyrest.cmake")
include("cmake/Sphinx.cmake")

if (DOXYGEN_FOUND AND DOXYREST_FOUND AND SPHINX_FOUND)
    add_custom_target(doc DEPENDS doc_doxygen doc_doxyrest doc_sphinx)
endif()
