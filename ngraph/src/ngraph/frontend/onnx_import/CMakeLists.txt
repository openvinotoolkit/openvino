# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
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

set(ONNX_OPSET_VERSION 11 CACHE INTERNAL "Supported version of ONNX operator set")

add_library(onnx_importer SHARED
        core/node.cpp
        core/node.hpp
        core/operator_set.hpp
        onnx.cpp
        onnx.hpp
        onnx_utils.hpp
        onnx_utils.cpp
        core/attribute.cpp
        core/attribute.hpp
        core/graph.cpp
        core/graph.hpp
        core/graph_cache.cpp
        core/graph_cache.hpp
        core/model.cpp
        core/model.hpp
        core/node.hpp
        core/null_node.cpp
        core/null_node.hpp
        core/operator_set.hpp
        core/tensor.hpp
        core/value_info.hpp
        default_opset.hpp
        exceptions.cpp
        exceptions.hpp
        op/acos.hpp
        op/acosh.cpp
        op/acosh.hpp
        op/add.hpp
        op/add.cpp
        op/and.hpp
        op/argmax.cpp
        op/argmax.hpp
        op/argmin.cpp
        op/argmin.hpp
        op/asin.hpp
        op/asinh.cpp
        op/asinh.hpp
        op/atan.hpp
        op/atanh.cpp
        op/atanh.hpp
        op/average_pool.cpp
        op/average_pool.hpp
        op/batch_norm.cpp
        op/batch_norm.hpp
        op/cast.cpp
        op/cast.hpp
        op/ceil.hpp
        op/clip.cpp
        op/clip.hpp
        op/cos.cpp
        op/cos.hpp
        op/cosh.cpp
        op/cosh.hpp
        op/concat.cpp
        op/concat.hpp
        op/constant.cpp
        op/constant.hpp
        op/constant_of_shape.cpp
        op/constant_of_shape.hpp
        op/conv.cpp
        op/conv.hpp
        op/conv_integer.cpp
        op/conv_integer.hpp
        op/conv_transpose.cpp
        op/conv_transpose.hpp
        op/cum_sum.cpp
        op/cum_sum.hpp
        op/depth_to_space.cpp
        op/depth_to_space.hpp
        op/dequantize_linear.cpp
        op/dequantize_linear.hpp
        op/div.hpp
        op/dropout.hpp
        op/elu.cpp
        op/elu.hpp
        op/equal.hpp
        op/erf.hpp
        op/exp.hpp
        op/expand.hpp
        op/expand.cpp
        op/eye_like.cpp
        op/eye_like.hpp
        op/flatten.cpp
        op/flatten.hpp
        op/floor.hpp
        op/gather.hpp
        op/gather_nd.hpp
        op/gather_nd.cpp
        op/gemm.cpp
        op/gemm.hpp
        op/global_average_pool.cpp
        op/global_average_pool.hpp
        op/global_max_pool.cpp
        op/global_max_pool.hpp
        op/greater.hpp
        op/gru.cpp
        op/gru.hpp
        op/hard_sigmoid.cpp
        op/hard_sigmoid.hpp
        op/hardmax.cpp
        op/hardmax.hpp
        op/identity.hpp
        op/instance_norm.cpp
        op/instance_norm.hpp
        op/leaky_relu.cpp
        op/leaky_relu.hpp
        op/less.hpp
        op/log.cpp
        op/log.hpp
        op/log_softmax.cpp
        op/log_softmax.hpp
        op/loop.cpp
        op/loop.hpp
        op/lp_norm.cpp
        op/lp_norm.hpp
        op/lp_pool.cpp
        op/lp_pool.hpp
        op/lrn.cpp
        op/lrn.hpp
        op/lstm.cpp
        op/lstm.hpp
        op/matmul.hpp
        op/matmul_integer.cpp
        op/matmul_integer.hpp
        op/max_pool.cpp
        op/max_pool.hpp
        op/max.hpp
        op/mean.cpp
        op/mean.hpp
        op/mean_variance_normalization.cpp
        op/mean_variance_normalization.hpp
        op/min.hpp
        op/mod.hpp
        op/mod.cpp
        op/mul.hpp
        op/neg.hpp
        op/not.hpp
        op/non_max_suppression.cpp
        op/non_max_suppression.hpp
        op/non_zero.cpp
        op/non_zero.hpp
        op/onehot.cpp
        op/onehot.hpp
        op/or.hpp
        op/pad.cpp
        op/pad.hpp
        op/pow.hpp
        op/prelu.cpp
        op/prelu.hpp
        op/qlinear_matmul.cpp
        op/qlinear_matmul.hpp
        op/quant_conv.cpp
        op/quant_conv.hpp
        op/quantize_linear.cpp
        op/quantize_linear.hpp
        op/range.cpp
        op/range.hpp
        op/reciprocal.cpp
        op/reciprocal.hpp
        op/reduce.cpp
        op/reduce.hpp
        op/relu.hpp
        op/reshape.cpp
        op/reshape.hpp
        op/resize.cpp
        op/resize.hpp
        op/reverse_sequence.cpp
        op/reverse_sequence.hpp
        op/rnn.cpp
        op/rnn.hpp
        op/roi_align.cpp
        op/roi_align.hpp
        op/round.cpp
        op/round.hpp
        op/scatter_nd.cpp
        op/scatter_nd.hpp
        op/scatter_elements.cpp
        op/scatter_elements.hpp
        op/selu.cpp
        op/selu.hpp
        op/shape.hpp
        op/shape.cpp
        op/shrink.hpp
        op/shrink.cpp
        op/sigmoid.hpp
        op/sign.hpp
        op/sin.hpp
        op/sinh.hpp
        op/size.cpp
        op/size.hpp
        op/slice.cpp
        op/slice.hpp
        op/softmax.cpp
        op/softmax.hpp
        op/softplus.cpp
        op/softplus.hpp
        op/softsign.cpp
        op/softsign.hpp
        op/space_to_depth.cpp
        op/space_to_depth.hpp
        op/split.cpp
        op/split.hpp
        op/sqrt.hpp
        op/squeeze.cpp
        op/squeeze.hpp
        op/sub.hpp
        op/sum.hpp
        op/tan.hpp
        op/tanh.hpp
        op/thresholded_relu.cpp
        op/thresholded_relu.hpp
        op/tile.cpp
        op/tile.hpp
        op/topk.cpp
        op/topk.hpp
        op/transpose.cpp
        op/transpose.hpp
        op/unsqueeze.cpp
        op/unsqueeze.hpp
        op/upsample.cpp
        op/upsample.hpp
        op/where.hpp
        op/xor.hpp
        ops_bridge.cpp
        ops_bridge.hpp
        utils/arg_min_max_factory.cpp
        utils/arg_min_max_factory.hpp
        utils/common.cpp
        utils/common.hpp
        utils/convpool.cpp
        utils/convpool.hpp
        utils/onnx_importer_visibility.hpp
        utils/pooling_factory.cpp
        utils/pooling_factory.hpp
        utils/recurrent.cpp
        utils/recurrent.hpp
        utils/provenance_tag.cpp
        utils/provenance_tag.hpp
        utils/reduction.cpp
        utils/reduction.hpp
        utils/reshape.cpp
        utils/reshape.hpp
        utils/variadic.hpp)

set(ONNX_IMPORT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")

target_link_libraries(onnx_importer PRIVATE onnx onnx_proto ${Protobuf_LIBRARIES})
target_link_libraries(onnx_importer PUBLIC ngraph)

set_target_properties(onnx_importer PROPERTIES
                      CXX_VISIBILITY_PRESET hidden
                      C_VISIBILITY_PRESET hidden
                      VISIBILITY_INLINES_HIDDEN ON
                      POSITION_INDEPENDENT_CODE ON)
target_include_directories(onnx_importer SYSTEM PUBLIC $<BUILD_INTERFACE:${ONNX_IMPORT_INCLUDE_DIR}>
                                                       $<INSTALL_INTERFACE:include/ngraph/frontend/onnx_import>)
target_include_directories(onnx_importer SYSTEM PRIVATE ${NGRAPH_INCLUDE_PATH}
        ${ONNX_INCLUDE_DIR} ${ONNX_PROTO_INCLUDE_DIR} ${Protobuf_INCLUDE_DIRS})

target_compile_definitions(onnx_importer PRIVATE ONNX_OPSET_VERSION=${ONNX_OPSET_VERSION})

if(NGRAPH_USE_PROTOBUF_LITE)
    target_compile_definitions(onnx_importer PRIVATE NGRAPH_USE_PROTOBUF_LITE)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(onnx_importer PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-invalid-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated PUBLIC -Wno-undefined-func-template)
endif()

target_compile_definitions(onnx_importer PRIVATE ONNX_IMPORTER_DLL_EXPORTS)

install(TARGETS onnx_importer EXPORT ngraphTargets
        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)
