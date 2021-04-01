# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
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
# ******************************************************************************

function(MODE_APPLY_FILE PATH)
    execute_process(COMMAND git update-index --add --chmod=-x ${PATH}
        OUTPUT_VARIABLE RESULT
        ERROR_QUIET)
endfunction()

set(DIRECTORIES_OF_INTEREST
    src
    doc
    test
    python/pyngraph
)

foreach(DIRECTORY ${DIRECTORIES_OF_INTEREST})
    set(DIR "${NGRAPH_SOURCE_DIR}/${DIRECTORY}/*.?pp")
    file(GLOB_RECURSE XPP_FILES ${DIR})
    foreach(FILE ${XPP_FILES})
        mode_apply_file(${FILE})
    endforeach(FILE)
endforeach(DIRECTORY)
