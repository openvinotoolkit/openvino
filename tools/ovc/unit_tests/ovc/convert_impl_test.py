import argparse

from openvino.tools.ovc.convert_impl import arguments_post_parsing


def test_arguments_post_parsing_keeps_tuple_outputs_separate():
    argv = argparse.Namespace(
        is_python_api_used=True,
        input=None,
        output=("output_1", "output_2"),
        placeholder_shapes=None,
    )

    arguments_post_parsing(argv)

    assert argv.output == ["output_1", "output_2"]
