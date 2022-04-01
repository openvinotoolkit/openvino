import numpy as np
from openvino.runtime import Layout, PartialShape, Dimension, Shape

from openvino.tools.mo.convert import input_to_str, InputCutInfo, LayoutMap, mean_scale_value_to_str, \
    transform_param_to_str, input_shape_to_str, str_list_to_str, source_target_layout_to_str, layout_param_to_str
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry


class TestConvertingConvertArgumentsToString(UnitTestWithMockedTelemetry):
    def test_input_to_str(self):
        inp1 = InputCutInfo("data:0")
        self.assertTrue(input_to_str(inp1) == "data:0")

        inp2 = InputCutInfo("data:0", [1, 3, 100, 100])
        self.assertTrue(input_to_str(inp2) == "data:0[1 3 100 100]")

        inp3 = InputCutInfo("data:0", type=np.int32)
        self.assertTrue(input_to_str(inp3) == "data:0{i32}")

        inp4 = InputCutInfo("data:0", value=[2, 4, 5])
        self.assertTrue(input_to_str(inp4) == "data:0->[2 4 5]")

        inp5 = InputCutInfo("data:0", [1, 3, 100, 100], np.uint8)
        self.assertTrue(input_to_str(inp5) == "data:0[1 3 100 100]{u8}")

        inp6 = InputCutInfo("data:0", [2, 5, 7], value=[1, 2, 3, 4, 5])
        self.assertTrue(input_to_str(inp6) == "data:0[2 5 7]->[1 2 3 4 5]")

        inp7 = InputCutInfo("0:data1", type=np.float64, value=[1.6, 7.2, 5.66])
        self.assertTrue(input_to_str(inp7) == "0:data1{f64}->[1.6 7.2 5.66]")

        inp8 = InputCutInfo("data2", [4, 5, 6], np.int64, [5, 4, 3, 2, 1])
        self.assertTrue(input_to_str(inp8) == "data2[4 5 6]{i64}->[5 4 3 2 1]")

        inp = [inp6, inp7, inp8]
        self.assertTrue(input_to_str(inp) == "data:0[2 5 7]->[1 2 3 4 5],"
                                             "0:data1{f64}->[1.6 7.2 5.66],"
                                             "data2[4 5 6]{i64}->[5 4 3 2 1]")

        inp = ["data:0[2 5 7]->[1 2 3 4 5]", "0:data1{f64}->[1.6 7.2 5.66]", "data2[4 5 6]{i64}->[5 4 3 2 1]"]
        self.assertTrue(input_to_str(inp) == "data:0[2 5 7]->[1 2 3 4 5],"
                                             "0:data1{f64}->[1.6 7.2 5.66],"
                                             "data2[4 5 6]{i64}->[5 4 3 2 1]")

        inp9 = InputCutInfo("data1", PartialShape([Dimension(-1), Dimension(2, -1),
                                                    Dimension(-1, 10), 100, Dimension(2, 12)]))
        self.assertTrue(input_to_str(inp9) == "data1[? 2.. ..10 100 2..12]")

        inp10 = InputCutInfo("data2", [Dimension(-1), Dimension(2, -1),
                                        Dimension(-1, 10), 100, Dimension(2, 12)], np.uint8)
        self.assertTrue(input_to_str(inp10) == "data2[? 2.. ..10 100 2..12]{u8}")

        inp11 = InputCutInfo("data3", Shape([4, 5, 6]), np.int64, [5, 4, 3, 2, 1])
        self.assertTrue(input_to_str(inp11) == "data3[4 5 6]{i64}->[5 4 3 2 1]")

        inp12 = InputCutInfo("data4", PartialShape.dynamic())
        self.assertTrue(input_to_str(inp12) == "data4[...]")

        inp = [inp9, inp10, inp11, inp12]
        self.assertTrue(input_to_str(inp) == "data1[? 2.. ..10 100 2..12],"
                                             "data2[? 2.. ..10 100 2..12]{u8},"
                                             "data3[4 5 6]{i64}->[5 4 3 2 1],"
                                             "data4[...]")

    def test_mean_scale_value_to_str(self):
        values = [0.5, 1.3, 0.67]
        self.assertTrue(mean_scale_value_to_str(values) == "[0.5,1.3,0.67]")

        values = {"input": [0.5, 1.3, 0.67]}
        self.assertTrue(mean_scale_value_to_str(values) == "input[0.5,1.3,0.67]")

        values = {"input1": [0.5, 1.3, 0.67], "input2": [4.2, 6.7, 3.15], "input3": [0.757, 4.6, 7.3]}
        self.assertTrue(mean_scale_value_to_str(values) ==
                        "input1[0.5,1.3,0.67],input2[4.2,6.7,3.15],input3[0.757,4.6,7.3]")

    def test_transform_param_to_str(self):
        transform = 'MakeStateful'
        self.assertTrue(transform_param_to_str(transform) == "MakeStateful")

        transform1 = ('LowLatency2', {'use_const_initializer': False})
        self.assertTrue(transform_param_to_str(transform1) ==
                        "LowLatency2[use_const_initializer=False]")

        transform2 = ('MakeStateful', {'param_res_names': {
            'input_name_1': 'output_name_1', 'input_name_2': 'output_name_2'}})
        self.assertTrue(transform_param_to_str(transform2) ==
                        "MakeStateful[param_res_names={\'input_name_1\':\'output_name_1\',"
                        "\'input_name_2\':\'output_name_2\'}]")

        transform = [transform1, transform2]

        self.assertTrue(transform_param_to_str(transform) == "LowLatency2[use_const_initializer=False],"
                                                             "MakeStateful[param_res_names={"
                                                             "\'input_name_1\':\'output_name_1\',"
                                                             "\'input_name_2\':\'output_name_2\'}]")

    def test_input_shape_to_str(self):
        input_shape1 = [1, 3, 100, 100]
        self.assertTrue(input_shape_to_str(input_shape1) == "[1,3,100,100]")

        input_shape2 = PartialShape([1, 3, 100, 100])
        self.assertTrue(input_shape_to_str(input_shape2) == "[1,3,100,100]")

        input_shape3 = PartialShape([Dimension(-1), Dimension(2, -1), Dimension(-1, 10), 100, Dimension(2, 12)])
        self.assertTrue(input_shape_to_str(input_shape3) == "[?,2..,..10,100,2..12]")

        input_shape4 = PartialShape.dynamic()
        self.assertTrue(input_shape_to_str(input_shape4) == "[...]")

        input_shape5 = Shape([1, 2, 3, 4])
        self.assertTrue(input_shape_to_str(input_shape5) == "[1,2,3,4]")

        input_shape6 = [Dimension(-1), Dimension(2, -1), Dimension(-1, 10), 100, Dimension(2, 12)]
        self.assertTrue(input_shape_to_str(input_shape6) == "[?,2..,..10,100,2..12]")

        input_shape = [input_shape1, input_shape2, input_shape3, input_shape4, input_shape5, input_shape6]
        self.assertTrue(input_shape_to_str(input_shape) == "[1,3,100,100],[1,3,100,100],[?,2..,..10,100,2..12],"
                                                           "[...],[1,2,3,4],[?,2..,..10,100,2..12]")

    def test_str_list_to_str(self):
        list_str = ["data1", "data2", "data3"]
        self.assertTrue(str_list_to_str(list_str) == "data1,data2,data3")

        list_str = "data1"
        self.assertTrue(str_list_to_str(list_str) == "data1")

    def test_source_target_layout_to_str(self):
        layout = {"input1":Layout("nhwc"), "input2":Layout("n??"), "input3":"nchw"}
        self.assertTrue(source_target_layout_to_str(layout) == "input1([N,H,W,C]),input2([N,?,?]),input3(nchw)")

    def test_layout_param_to_str_to_str(self):
        layout = {"input1": Layout("nhwc"), "input2": Layout("n??"), "input3": "nchw"}
        self.assertTrue(layout_param_to_str(layout) == "input1([N,H,W,C]),input2([N,?,?]),input3(nchw)")

        layout_map1 = LayoutMap(source_layout=Layout("n??"))
        layout_map2 = LayoutMap(source_layout=Layout("nhwc"), target_layout=("nchw"))
        layout_map3 = LayoutMap(source_layout="abc", target_layout="cab")

        layout = {"input1": layout_map1, "input2": layout_map2, "input3": layout_map3, "input4": Layout("nhwc"), "input5": "n?"}

        self.assertTrue(layout_param_to_str(layout) == "input1([N,?,?]),input2([N,H,W,C]->nchw),"
                                                       "input3(abc->cab),input4([N,H,W,C]),input5(n?)")


