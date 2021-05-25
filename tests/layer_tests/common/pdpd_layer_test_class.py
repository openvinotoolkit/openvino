from pathlib import Path

from common.layer_test_class import CommonLayerTest


class CommonPaddlePaddleLayerTest(CommonLayerTest):
    """Base class for PaddlePaddle layer tests"""
    @staticmethod
    def save_to_paddlepaddle(model_data, path_to_saved_pdpd_model):
        from paddle import fluid

        executor = fluid.Executor(fluid.CPUPlace())

        fluid.io.save_inference_model(
            dirname=path_to_saved_pdpd_model,
            feeded_var_names=model_data["input_layers_names"],
            target_vars=model_data["output_layers"],
            executor=executor,
            main_program=model_data["inference_program"]
        )
        model_path = Path(path_to_saved_pdpd_model) / "__model__"
        assert model_path.is_file(), f"PaddlePaddle haven't been saved here: {path_to_saved_pdpd_model}"

        return str(model_path)


    def produce_model_path(self, framework_model, save_path):

        return self.save_to_paddlepaddle(framework_model, save_path)


    def get_framework_results(self, inputs_dict, model_path):
        """Return PaddlePaddle model reference results."""
        from paddle import enable_static, fluid

        executor = fluid.Executor(fluid.CPUPlace())

        enable_static()
        inference_program, _, output_layers = fluid.io.load_inference_model(
            executor=executor,
            dirname=model_path
        )
        out = executor.run(inference_program, feed=inputs_dict, fetch_list=output_layers, return_numpy=False)
        res = dict(zip(map(lambda layer: layer.name, output_layers), out))

        return res
