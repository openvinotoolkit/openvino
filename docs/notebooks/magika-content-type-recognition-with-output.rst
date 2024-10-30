Magika: AI powered fast and efficient file type identification using OpenVINO
=============================================================================

Magika is a novel AI powered file type detection tool that relies on the
recent advance of deep learning to provide accurate detection. Under the
hood, Magika employs a custom, highly optimized model that only weighs
about 1MB, and enables precise file identification within milliseconds,
even when running on a single CPU.

Why identifying file type is difficult
--------------------------------------

Since the early days of computing, accurately detecting file types has
been crucial in determining how to process files. Linux comes equipped
with ``libmagic`` and the ``file`` utility, which have served as the de
facto standard for file type identification for over 50 years. Today web
browsers, code editors, and countless other software rely on file-type
detection to decide how to properly render a file. For example, modern
code editors use file-type detection to choose which syntax coloring
scheme to use as the developer starts typing in a new file.

Accurate file-type detection is a notoriously difficult problem because
each file format has a different structure, or no structure at all. This
is particularly challenging for textual formats and programming
languages as they have very similar constructs. So far, ``libmagic`` and
most other file-type-identification software have been relying on a
handcrafted collection of heuristics and custom rules to detect each
file format.

This manual approach is both time consuming and error prone as it is
hard for humans to create generalized rules by hand. In particular for
security applications, creating dependable detection is especially
challenging as attackers are constantly attempting to confuse detection
with adversarially-crafted payloads.

To address this issue and provide fast and accurate file-type detection
Magika was developed. More details about approach and model can be found
in original `repo <https://github.com/google/magika>`__ and `Google’s
blog
post <https://opensource.googleblog.com/2024/02/magika-ai-powered-fast-and-efficient-file-type-identification.html>`__.

In this tutorial we consider how to bring OpenVINO power into Magika.

**Table of contents:**

-  `Prerequisites <#prerequisites>`__
-  `Define model loading class <#define-model-loading-class>`__
-  `Run OpenVINO model inference <#run-openvino-model-inference>`__

   -  `Select device <#select-device>`__
   -  `Create model <#create-model>`__
   -  `Run inference on bytes input <#run-inference-on-bytes-input>`__
   -  `Run inference on file input <#run-inference-on-file-input>`__

-  `Interactive demo <#interactive-demo>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



.. code:: ipython3

    %pip install -q magika "openvino>=2024.1.0" "gradio>=4.19"


.. parsed-literal::

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    supervision 0.24.0 requires numpy<1.23.3,>=1.21.2; python_full_version <= "3.10.0", but you have numpy 1.24.4 which is incompatible.
    tensorflow 2.12.0 requires numpy<1.24,>=1.22, but you have numpy 1.24.4 which is incompatible.
    Note: you may need to restart the kernel to use updated packages.


Define model loading class
--------------------------



At inference time Magika uses ONNX as an inference engine to ensure
files are identified in a matter of milliseconds, almost as fast as a
non-AI tool even on CPU. The code below extending original Magika
inference class with OpenVINO API. The provided code is fully compatible
with original `Magika Python
API <https://github.com/google/magika/blob/main/docs/python.md>`__.

.. code:: ipython3

    import time
    from pathlib import Path
    from functools import partial
    from typing import List, Tuple, Optional, Dict

    from magika import Magika
    from magika.types import ModelFeatures, ModelOutput, MagikaResult
    from magika.prediction_mode import PredictionMode
    import numpy.typing as npt
    import numpy as np

    import openvino as ov


    class OVMagika(Magika):
        def __init__(
            self,
            model_dir: Optional[Path] = None,
            prediction_mode: PredictionMode = PredictionMode.HIGH_CONFIDENCE,
            no_dereference: bool = False,
            verbose: bool = False,
            debug: bool = False,
            use_colors: bool = False,
            device="CPU",
        ) -> None:
            self._device = device
            super().__init__(model_dir, prediction_mode, no_dereference, verbose, debug, use_colors)

        def _init_onnx_session(self):
            # overload model loading using OpenVINO
            start_time = time.time()
            core = ov.Core()
            ov_model = core.compile_model(self._model_path, self._device.upper())
            elapsed_time = 1000 * (time.time() - start_time)
            self._log.debug(f'ONNX DL model "{self._model_path}" loaded in {elapsed_time:.03f} ms on {self._device}')
            return ov_model

        def _get_raw_predictions(self, features: List[Tuple[Path, ModelFeatures]]) -> npt.NDArray:
            """
            Given a list of (path, features), return a (files_num, features_size)
            matrix encoding the predictions.
            """

            dataset_format = self._model_config["train_dataset_info"]["dataset_format"]
            assert dataset_format == "int-concat/one-hot"
            start_time = time.time()
            X_bytes = []
            for _, fs in features:
                sample_bytes = []
                if self._input_sizes["beg"] > 0:
                    sample_bytes.extend(fs.beg[: self._input_sizes["beg"]])
                if self._input_sizes["mid"] > 0:
                    sample_bytes.extend(fs.mid[: self._input_sizes["mid"]])
                if self._input_sizes["end"] > 0:
                    sample_bytes.extend(fs.end[-self._input_sizes["end"] :])
                X_bytes.append(sample_bytes)
            X = np.array(X_bytes).astype(np.float32)
            elapsed_time = time.time() - start_time
            self._log.debug(f"DL input prepared in {elapsed_time:.03f} seconds")

            start_time = time.time()
            raw_predictions_list = []
            samples_num = X.shape[0]

            max_internal_batch_size = 1000
            batches_num = samples_num // max_internal_batch_size
            if samples_num % max_internal_batch_size != 0:
                batches_num += 1

            for batch_idx in range(batches_num):
                self._log.debug(f"Getting raw predictions for (internal) batch {batch_idx+1}/{batches_num}")
                start_idx = batch_idx * max_internal_batch_size
                end_idx = min((batch_idx + 1) * max_internal_batch_size, samples_num)
                batch_raw_predictions = self._onnx_session({"bytes": X[start_idx:end_idx, :]})["target_label"]
                raw_predictions_list.append(batch_raw_predictions)
            elapsed_time = time.time() - start_time
            self._log.debug(f"DL raw prediction in {elapsed_time:.03f} seconds")
            return np.concatenate(raw_predictions_list)

        def _get_topk_model_outputs_from_features(self, all_features: List[Tuple[Path, ModelFeatures]], k: int = 5) -> List[Tuple[Path, List[ModelOutput]]]:
            """
            Helper function for getting top k the highest ranked model results for each feature
            """
            raw_preds = self._get_raw_predictions(all_features)
            top_preds_idxs = np.argsort(raw_preds, axis=1)[:, -k:][:, ::-1]
            scores = [raw_preds[i, idx] for i, idx in enumerate(top_preds_idxs)]
            results = []
            for (path, _), scores, top_idxes in zip(all_features, raw_preds, top_preds_idxs):
                model_outputs_for_path = []
                for idx in top_idxes:
                    ct_label = self._target_labels_space_np[idx]
                    score = scores[idx]
                    model_outputs_for_path.append(ModelOutput(ct_label=ct_label, score=float(score)))
                results.append((path, model_outputs_for_path))
            return results

        def _get_results_from_features_topk(self, all_features: List[Tuple[Path, ModelFeatures]], top_k=5) -> Dict[str, MagikaResult]:
            """
            Helper function for getting top k the highest ranked model results for each feature
            """
            # We now do inference for those files that need it.

            if len(all_features) == 0:
                # nothing to be done
                return {}

            outputs: Dict[str, MagikaResult] = {}

            for path, model_output in self._get_topk_model_outputs_from_features(all_features, top_k):
                # In additional to the content type label from the DL model, we
                # also allow for other logic to overwrite such result. For
                # debugging and information purposes, the JSON output stores
                # both the raw DL model output and the final output we return to
                # the user.
                results = []
                for out in model_output:
                    output_ct_label = self._get_output_ct_label_from_dl_result(out.ct_label, out.score)

                    results.append(
                        self._get_result_from_labels_and_score(
                            path,
                            dl_ct_label=out.ct_label,
                            output_ct_label=output_ct_label,
                            score=out.score,
                        )
                    )
                outputs[str(path)] = results

            return outputs

        def identify_bytes_topk(self, content: bytes, top_k=5) -> MagikaResult:
            # Helper function for getting topk results from bytes
            _get_results_from_features = self._get_results_from_features
            self._get_results_from_features = partial(self._get_results_from_features_topk, top_k=top_k)
            result = super().identify_bytes(content)
            self._get_results_from_features = _get_results_from_features
            return result

Run OpenVINO model inference
----------------------------



Now let’s check model inference result.

Select device
~~~~~~~~~~~~~



For starting work, please, select one of represented devices from
dropdown list.

.. code:: ipython3

    import requests

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)

    from notebook_utils import device_widget

    device = device_widget()

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Create model
~~~~~~~~~~~~



As we discussed above, our OpenVINO extended ``OVMagika`` class has the
same API like original one. Let’s try to create interface instance and
launch it on different input formats

.. code:: ipython3

    ov_magika = OVMagika(device=device.value)

Run inference on bytes input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    result = ov_magika.identify_bytes(b"# Example\nThis is an example of markdown!")
    print(f"Content type: {result.output.ct_label} - {result.output.score * 100:.4}%")


.. parsed-literal::

    Content type: markdown - 99.29%


Run inference on file input
~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    import requests

    input_file = Path("./README.md")
    if not input_file.exists():
        r = requests.get("https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/README.md")
        with open("README.md", "w") as f:
            f.write(r.text)
    result = ov_magika.identify_path(input_file)
    print(f"Content type: {result.output.ct_label} - {result.output.score * 100:.4}%")


.. parsed-literal::

    Content type: markdown - 100.0%


Interactive demo
----------------



Now, you can try model on own files. Upload file into input file window,
click submit button and look on predicted file types.

.. code:: ipython3

    import gradio as gr


    def classify(file_path):
        """Classify file using classes listing.
        Args:
            file_path): path to input file
        Returns:
            (dict): Mapping between class labels and class probabilities.
        """
        results = ov_magika.identify_bytes_topk(file_path)

        return {result.dl.ct_label: float(result.output.score) for result in results}


    demo = gr.Interface(
        fn=classify,
        inputs=[
            gr.File(label="Input file", type="binary"),
        ],
        outputs=gr.Label(label="Result"),
        examples=[["./README.md"]],
        allow_flagging="never",
    )
    try:
        demo.launch(debug=False)
    except Exception:
        demo.launch(share=True, debug=False)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860

    To create a public link, set `share=True` in `launch()`.








.. code:: ipython3

    # please uncomment and run this cell for stopping gradio interface
    # demo.close()
