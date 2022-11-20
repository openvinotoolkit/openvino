# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from openvino.tools.mo.utils.error import Error

try:
    import tensorflow.compat.v1 as tf
    from google.protobuf import text_format
    from tensorboard.compat.proto import event_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
except ImportError:
    raise Error(
        'For generation of TensorBoard logs for OpenVINO Model,'
        'nstall required dependencies: pip install tensorflow protobuf tensorboard')

from openvino.runtime import Model  # pylint: disable=no-name-in-module,import-error


class SummaryWriter(object):
    """
    Writes TensorBoard logs for OpenVINO model object
    """

    def __init__(
            self,
            log_dir=None,
            max_queue=10,
            flush_secs=120,
            filename_suffix="",
    ):
        if not log_dir:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime("%b%d_%H-%M-%S")
            log_dir = os.path.join(
                "runs", current_time + "_" + socket.gethostname()
            )
        # initialize the event writer for OpenVINO Model
        self.event_writer = EventFileWriter(log_dir, max_queue, flush_secs, filename_suffix)

    def add_graph(
            self, ov_model: Model
    ):
        """
        Add OpenVINO model data to summary.
        """
        # write ov::Model in protobuf format
        graph_def_str = ""
        for op_node in ov_model.get_ordered_ops():
            op_type = op_node.get_type_name()
            node_name = op_node.get_friendly_name()
            node_def_str = "  name: \"{}\"\n".format(node_name)
            node_def_str += "  op: \"{}\"\n".format(op_type)
            for ind, input_value in enumerate(op_node.input_values()):
                if op_node.get_input_size() == 1:
                    node_def_str += "  input: \"{}\"\n".format(input_value.get_node().get_friendly_name())
                else:
                    node_def_str += "  input: \"{}:{}\"\n".format(input_value.get_node().get_friendly_name(), ind)
            # TODO: add other attributes, not only _output_shapes
            # generate a list of output shapes
            attr_def_str = "  attr {\n"
            attr_def_str += "    key: \"_output_shapes\"\n"
            attr_def_str += "    value {\n"
            attr_def_str += "      list {\n"
            for ind in range(0, op_node.get_output_size()):
                partial_shape = op_node.get_output_partial_shape(ind)
                if partial_shape.rank.is_static:
                    attr_def_str += "        shape {\n"
                    for dim in partial_shape:
                        dim_value = -1
                        if dim.is_static:
                            dim_value = dim.get_length()
                        attr_def_str += "          dim {\n"
                        attr_def_str += "            size: {}\n".format(dim_value)
                        attr_def_str += "          }\n"
                    attr_def_str += "        }\n"
                else:
                    attr_def_str += "        shape {\n"
                    attr_def_str += "        }\n"

            attr_def_str += "      }\n"
            attr_def_str += "    }\n"
            attr_def_str += "  }\n"
            node_def_str = "node {\n" + node_def_str + attr_def_str
            node_def_str += "}\n"
            graph_def_str += node_def_str
        graph_def = tf.GraphDef()
        text_format.Merge(graph_def_str, graph_def)
        event_writer = EventFileWriter(".", 10, 120, "")
        event = event_pb2.Event(graph_def=graph_def.SerializeToString())
        event_writer.add_event(event)

    def flush(self):
        """
        Flushes the event file to disk.
        """
        self.event_writer.flush()

    def close(self):
        self.event_writer.flush()
        self.event_writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
