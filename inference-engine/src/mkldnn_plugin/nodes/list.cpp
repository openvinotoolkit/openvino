// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/list.hpp"

#include "mkldnn_reference_node.h"
#include "mkldnn_shapeof.h"
#include "mkldnn_batch_to_space_node.h"
#include "mkldnn_multiclass_nms.hpp"
#include "mkldnn_adaptive_pooling.h"
#include "mkldnn_conv_node.h"
#include "mkldnn_roi_align_node.h"
#include "mkldnn_lrn_node.h"
#include "mkldnn_generic_node.h"
#include "mkldnn_experimental_detectron_roifeatureextractor_node.h"
#include "mkldnn_eltwise_node.h"
#include "mkldnn_reorg_yolo_node.h"
#include "mkldnn_pooling_node.h"
#include "mkldnn_transpose_node.h"
#include "mkldnn_grn_node.h"
#include "mkldnn_interpolate_node.h"
#include "mkldnn_experimental_detectron_detection_output_node.h"
#include "mkldnn_roll_node.h"
#include "mkldnn_fake_quantize_node.h"
#include "mkldnn_embedding_segments_sum_node.h"
#include "mkldnn_region_yolo_node.h"
#include "mkldnn_matmul_node.h"
#include "mkldnn_detection_output_node.h"
#include "mkldnn_reverse_sequence_node.h"
#include "mkldnn_pad_node.h"
#include "mkldnn_ctc_greedy_decoder_seq_len_node.h"
#include "mkldnn_reshape_node.h"
#include "mkldnn_psroi_pooling_node.h"
#include "mkldnn_memory_node.hpp"
#include "mkldnn_bin_conv_node.h"
#include "mkldnn_gather_elements_node.h"
#include "mkldnn_experimental_detectron_priorgridgenerator_node.h"
#include "mkldnn_tile_node.h"
#include "mkldnn_math_node.h"
#include "mkldnn_normalize_node.h"
#include "mkldnn_proposal_node.h"
#include "mkldnn_tensoriterator_node.h"
#include "mkldnn_fullyconnected_node.h"
#include "mkldnn_extract_image_patches_node.h"
#include "mkldnn_ctc_loss_node.h"
#include "mkldnn_reorder_node.h"
#include "mkldnn_gather_nd_node.h"
#include "mkldnn_shuffle_channels_node.h"
#include "mkldnn_bucketize_node.h"
#include "mkldnn_space_to_depth_node.h"
#include "mkldnn_concat_node.h"
#include "mkldnn_softmax_node.h"
#include "mkldnn_space_to_batch_node.h"
#include "mkldnn_select_node.h"
#include "mkldnn_topk_node.h"
#include "mkldnn_broadcast_node.h"
#include "mkldnn_matrix_nms_node.h"
#include "mkldnn_mvn_node.h"
#include "mkldnn_gather_node.h"
#include "mkldnn_scatter_update_node.h"
#include "mkldnn_gather_tree_node.h"
#include "mkldnn_def_conv_node.h"
#include "mkldnn_embedding_bag_offset_sum_node.h"
#include "mkldnn_deconv_node.h"
#include "mkldnn_roi_pooling_node.h"
#include "mkldnn_range_node.h"
#include "mkldnn_split_node.h"
#include "mkldnn_one_hot_node.h"
#include "mkldnn_log_softmax_node.h"
#include "mkldnn_strided_slice_node.h"
#include "mkldnn_dft_node.h"
#include "mkldnn_non_max_suppression_node.h"
#include "mkldnn_convert_node.h"
#include "mkldnn_rnn.h"
#include "mkldnn_experimental_detectron_topkrois_node.h"
#include "mkldnn_cum_sum_node.h"
#include "mkldnn_depth_to_space_node.h"
#include "mkldnn_input_node.h"
#include "mkldnn_experimental_detectron_generate_proposals_single_image_node.h"
#include "mkldnn_embedding_bag_packed_sum_node.h"
#include "mkldnn_reduce_node.h"
#include "mkldnn_if_node.h"
#include "mkldnn_ctc_greedy_decoder_node.h"

namespace MKLDNNPlugin {

#define FACTORY_DECLARATION(__prim, __type) \
    void __prim ## __type();

#define MKLDNN_NODE(__prim, __type) FACTORY_DECLARATION(__prim, __type)
# include "list_tbl.hpp"
#undef MKLDNN_NODE
#undef FACTORY_DECLARATION

} // namespace MKLDNNPlugin

#define FACTORY_INOVCATION(__prim, __type) \
    MKLDNNPlugin :: __prim ## __type();

InferenceEngine::Extensions::Cpu::MKLDNNExtensions::MKLDNNExtensions() : layersFactory("LayersFactory") {
#define MKLDNN_NODE(__prim, __type) FACTORY_INOVCATION(__prim, __type)
# include "list_tbl.hpp"
#undef MKLDNN_NODE
#undef FACTORY_INOVCATION
}
