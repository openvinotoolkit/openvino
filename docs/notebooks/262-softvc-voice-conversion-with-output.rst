SoftVC VITS Singing Voice Conversion and OpenVINO™
==================================================

This tutorial is based on `SoftVC VITS Singing Voice Conversion
project <https://github.com/svc-develop-team/so-vits-svc>`__. The
purpose of this project was to enable developers to have their beloved
anime characters perform singing tasks. The developers’ intention was to
focus solely on fictional characters and avoid any involvement of real
individuals, anything related to real individuals deviates from the
developer’s original intention.

The singing voice conversion model uses SoftVC content encoder to
extract speech features from the source audio. These feature vectors are
directly fed into `VITS <https://github.com/jaywalnut310/vits>`__
without the need for conversion to a text-based intermediate
representation. As a result, the pitch and intonations of the original
audio are preserved.

In this tutorial we will use the base model flow.

**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Use the original model to run an
   inference <#use-the-original-model-to-run-an-inference->`__
-  `Convert the original model to OpenVINO Intermediate Representation
   (IR)
   format <#convert-the-original-model-to-openvino-intermediate-representation-ir-format>`__
-  `Run the OpenVINO model <#run-the-openvino-model>`__
-  `Interactive inference <#interactive-inference>`__

Prerequisites
-------------

.. code:: ipython3

    %pip install -q --upgrade pip setuptools
    %pip install -q "openvino>=2023.2.0.dev20230922"
    !git clone https://github.com/svc-develop-team/so-vits-svc -b 4.1-Stable
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu  tqdm librosa torch torchaudio faiss-cpu gradio "numpy==1.23.5" "fairseq==0.12.2" praat-parselmouth

Download pretrained models and configs. We use a recommended encoder
`ContentVec <https://arxiv.org/abs/2204.09224>`__ and models from `a
collection of so-vits-svc-4.0 models made by the Pony Preservation
Project <https://huggingface.co/therealvul/so-vits-svc-4.0>`__ for
example. You can choose any other pretrained model from this or another
project or `prepare your
own <https://github.com/svc-develop-team/so-vits-svc#%EF%B8%8F-training>`__.

.. code:: ipython3

    # Fetch `notebook_utils` module
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    from notebook_utils import download_file
    
    # ContentVec
    download_file("https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt", "checkpoint_best_legacy_500.pt", directory="so-vits-svc/pretrain/")
    
    # pretrained models and configs from a collection of so-vits-svc-4.0 models. You can use other models. 
    download_file("https://huggingface.co/therealvul/so-vits-svc-4.0/resolve/main/Rainbow%20Dash%20(singing)/kmeans_10000.pt", "kmeans_10000.pt", directory="so-vits-svc/logs/44k/")
    download_file("https://huggingface.co/therealvul/so-vits-svc-4.0/resolve/main/Rainbow%20Dash%20(singing)/config.json", "config.json", directory="so-vits-svc/configs/")
    download_file("https://huggingface.co/therealvul/so-vits-svc-4.0/resolve/main/Rainbow%20Dash%20(singing)/G_30400.pth", "G_30400.pth", directory="so-vits-svc/logs/44k/")
    download_file("https://huggingface.co/therealvul/so-vits-svc-4.0/resolve/main/Rainbow%20Dash%20(singing)/D_30400.pth", "D_30400.pth", directory="so-vits-svc/logs/44k/")
    
    # a wav sample
    download_file("https://huggingface.co/datasets/santifiorino/spinetta/resolve/main/spinetta/000.wav", "000.wav", directory="so-vits-svc/raw/")

Use the original model to run an inference `⇧ <#table-of-content>`__
---------------------------------------------------------------------

Change directory to ``so-vits-svc`` in purpose not to brake internal
relative paths.

.. code:: ipython3

    %cd so-vits-svc

Define the Sovits Model.

.. code:: ipython3

    from inference.infer_tool import Svc
    
    model = Svc("logs/44k/G_30400.pth", "configs/config.json", device='cpu')

Define ``kwargs`` and make an inference.

.. code:: ipython3

    kwargs = {
        'raw_audio_path': 'raw/000.wav',  # path to a source audio 
        'spk': 'Rainbow Dash (singing)',  # speaker ID in which the source audio should be converted. 
        'tran': 0, 
        'slice_db': -40, 
        'cluster_infer_ratio': 0, 
        'auto_predict_f0': False, 
        'noice_scale': 0.4, 
    }
    
    audio = model.slice_inference(**kwargs)

And let compare the original audio with the result.

.. code:: ipython3

    import IPython.display as ipd
    
    # original
    ipd.Audio("raw/000.wav", rate=model.target_sample)

.. code:: ipython3

    # result
    ipd.Audio(audio, rate=model.target_sample)

Convert to OpenVINO IR model
----------------------------

Model components are PyTorch modules, that can be converted with
``ov.convert_model`` function directly. We also use ``ov.save_model``
function to serialize the result of conversion. ``Svc`` is not a model,
it runs model inference inside. In base scenario only ``SynthesizerTrn``
named ``net_g_ms`` is used. It is enough to convert only this model and
we should re-assign ``forward`` method on ``infer`` method for this
purpose.

``SynthesizerTrn`` uses several models inside it’s flow,
i.e. \ ``TextEncoder``, ``Generator``, ``ResidualCouplingBlock``, etc.,
but in our case OpenVINO allows to convert whole pipeline by one step
without need to look inside.

.. code:: ipython3

    import openvino as ov
    import torch
    from pathlib import Path
    
    
    dummy_c = torch.randn(1, 256, 813)
    dummy_f0 = torch.randn(1, 813)
    dummy_uv = torch.ones(1, 813)
    dummy_g = torch.tensor([[0]])
    model.net_g_ms.forward = model.net_g_ms.infer
    
    net_g_kwargs = {
        'c': dummy_c,
        'f0': dummy_f0,
        'uv': dummy_uv,
        'g': dummy_g,
        'noice_scale': torch.tensor(0.35),  # need to wrap numeric and boolean values for conversion
        'seed': torch.tensor(52468),
        'predict_f0': torch.tensor(False),
        'vol': torch.tensor(0)
    }
    core = ov.Core()
    
    
    net_g_model_xml_path = Path('models/ov_net_g_model.xml')
    
    if not net_g_model_xml_path.exists():
        converted_model = ov.convert_model(model.net_g_ms, example_input=net_g_kwargs)
        net_g_model_xml_path.parent.mkdir(parents=True, exist_ok=True)
        ov.save_model(converted_model, net_g_model_xml_path)

Run the OpenVINO model
----------------------

Select a device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    import ipywidgets as widgets
    import openvino as ov
    
    core = ov.Core()
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device

We should create a wrapper for ``net_g_ms`` model to keep it’s
interface. Then replace ``net_g_ms`` original model by the converted IR
model. We use ``ov.compile_model`` to make it ready to use for loading
on a device.

.. code:: ipython3

    class NetGModelWrapper:
        def __init__(self, net_g_model_xml_path):
            super().__init__()
            self.net_g_model = core.compile_model(net_g_model_xml_path, device.value)
            
        def infer(self, c, *, f0, uv, g, noice_scale=0.35, seed=52468, predict_f0=False, vol=None):
            if vol is None:  # None is not allowed as an input
                results = self.net_g_model((c, f0, uv, g, noice_scale, seed, predict_f0))
            else:
                results = self.net_g_model((c, f0, uv, g, noice_scale, seed, predict_f0, vol))
            
            return torch.from_numpy(results[0]), torch.from_numpy(results[1])
    
            
    model.net_g_ms = NetGModelWrapper(net_g_model_xml_path)
    audio = model.slice_inference(**kwargs)

Check result. Is it identical to that created by the original model.

.. code:: ipython3

    import IPython.display as ipd
    
    ipd.Audio(audio, rate=model.target_sample)

Interactive inference
---------------------

.. code:: ipython3

    import gradio as gr
    
    
    src_audio = gr.inputs.Audio(label="Source Audio", type='filepath')
    output_audio = gr.outputs.Audio(label="Output Audio", type='numpy')
    
    title = 'SoftVC VITS Singing Voice Conversion with Gradio'
    description = f'Gradio Demo for SoftVC VITS Singing Voice Conversion and OpenVINO™. Upload a source audio, then click the "Submit" button to inference. Audio sample rate should be {model.target_sample}'
    
    
    def infer(src_audio, tran, slice_db, noice_scale):
        kwargs["raw_audio_path"] = src_audio
        kwargs["tran"] = tran
        kwargs["slice_db"] = slice_db
        kwargs["noice_scale"] = noice_scale
     
        audio = model.slice_inference(**kwargs)
    
        return model.target_sample, audio
    
    
    demo = gr.Interface(
        infer, 
        [
            src_audio,
            gr.Slider(-100, 100, value=0, label="Pitch shift", step=1),
            gr.Slider(-80, -20, value=-30, label="Slice db", step=10, info="The default is -30, noisy audio can be -30, dry sound can be -50 to preserve breathing."),
            gr.Slider(0, 1, value=0.4, label="Noise scale", step=0.1, info="Noise level will affect pronunciation and sound quality, which is more metaphysical"),
        ],
        output_audio, 
        title=title,
        description=description, 
        examples=[['raw/000.wav', 0, -30, 0.4, False]]
    )
    
    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(share=True, debug=False)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
