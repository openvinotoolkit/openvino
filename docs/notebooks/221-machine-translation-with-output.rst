Machine translation demo
========================

This demo utilizes Intel’s pre-trained model that translates from
English to German. More information about the model can be found
`here <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/machine-translation-nar-en-de-0002/README.md>`__.

This model encodes sentences using the ``SentecePieceBPETokenizer`` from
HuggingFace. The tokenizer vocabulary is downloaded automatically with
the OMZ tool.

**Inputs** The model’s input is a sequence of up to 150 tokens with the
following structure: ``<s>`` + *tokenized sentence* + ``<s>`` +
``<pad>`` (``<pad>`` tokens pad the remaining blank spaces).

**Output** After the inference, we have a sequence of up to 200 tokens.
The structure is the same as the one for the input.

**Table of contents:**


-  `Downloading model <#downloading-model>`__
-  `Load and configure the
   model <#load-and-configure-the-model>`__
-  `Select inference device <#select-inference-device>`__
-  `Load tokenizers <#load-tokenizers>`__
-  `Perform translation <#perform-translation>`__
-  `Translate the sentence <#translate-the-sentence>`__

   -  `Test your translation <#test-your-translation>`__

.. code:: ipython3

    # # Install requirements
    %pip install -q "openvino>=2023.1.0"
    %pip install -q tokenizers


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Note: you may need to restart the kernel to use updated packages.
    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import time
    import sys
    import openvino as ov
    import numpy as np
    import itertools
    from pathlib import Path
    from tokenizers import SentencePieceBPETokenizer
    
    sys.path.append("../utils")
    from notebook_utils import download_file

Downloading model 
-----------------------------------------------------------

The following command will download the model to the current directory.
Make sure you have run ``pip install openvino-dev`` beforehand.

.. code:: ipython3

    base_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1"
    model_name = "machine-translation-nar-en-de-0002"
    precision = "FP32"
    model_base_dir = Path("model")
    model_base_dir.mkdir(exist_ok=True)
    model_path = model_base_dir / f"{model_name}.xml"
    src_tok_dir = model_base_dir / "tokenizer_src"
    target_tok_dir = model_base_dir / "tokenizer_tgt"
    src_tok_dir.mkdir(exist_ok=True)
    target_tok_dir.mkdir(exist_ok=True)
    
    download_file(base_url + f'/{model_name}/{precision}/{model_name}.xml', f"{model_name}.xml", model_base_dir)
    download_file(base_url + f'/{model_name}/{precision}/{model_name}.bin', f"{model_name}.bin", model_base_dir)
    download_file(f"{base_url}/{model_name}/tokenizer_src/merges.txt", "merges.txt", src_tok_dir)
    download_file(f"{base_url}/{model_name}/tokenizer_tgt/merges.txt", "merges.txt", target_tok_dir)
    download_file(f"{base_url}/{model_name}/tokenizer_src/vocab.json", "vocab.json", src_tok_dir)
    download_file(f"{base_url}/{model_name}/tokenizer_tgt/vocab.json", "vocab.json", target_tok_dir);



.. parsed-literal::

    model/machine-translation-nar-en-de-0002.xml:   0%|          | 0.00/825k [00:00<?, ?B/s]



.. parsed-literal::

    model/machine-translation-nar-en-de-0002.bin:   0%|          | 0.00/271M [00:00<?, ?B/s]



.. parsed-literal::

    model/tokenizer_src/merges.txt:   0%|          | 0.00/311k [00:00<?, ?B/s]



.. parsed-literal::

    model/tokenizer_tgt/merges.txt:   0%|          | 0.00/331k [00:00<?, ?B/s]



.. parsed-literal::

    model/tokenizer_src/vocab.json:   0%|          | 0.00/523k [00:00<?, ?B/s]



.. parsed-literal::

    model/tokenizer_tgt/vocab.json:   0%|          | 0.00/543k [00:00<?, ?B/s]


Load and configure the model 
----------------------------------------------------------------------

The model is now available in the ``model/`` folder. Below, we load and
configure its inputs and outputs.

.. code:: ipython3

    core = ov.Core()
    model = core.read_model(model_path)
    input_name = "tokens"
    output_name = "pred"
    model.output(output_name)
    max_tokens = model.input(input_name).shape[1]

Select inference device 
-----------------------------------------------------------------

select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    
    core = ov.Core()
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    compiled_model = core.compile_model(model, device.value)

Load tokenizers 
---------------------------------------------------------

NLP models usually take a list of tokens as standard input. A token is a
single word converted to some integer. To provide the proper input, we
need the vocabulary for such mapping. We use ``merges.txt`` to find out
what sequences of letters form a token. ``vocab.json`` specifies the
mapping between tokens and integers.

The input needs to be transformed into a token sequence the model
understands, and the output must be transformed into a sentence that is
human readable.

Initialize the tokenizer for the input ``src_tokenizer`` and the output
``tgt_tokenizer``.

.. code:: ipython3

    src_tokenizer = SentencePieceBPETokenizer.from_file(
        str(src_tok_dir / 'vocab.json'),
        str(src_tok_dir / 'merges.txt')
    )
    tgt_tokenizer = SentencePieceBPETokenizer.from_file(
        str(target_tok_dir / 'vocab.json'),
        str(target_tok_dir / 'merges.txt')
    )

Perform translation 
-------------------------------------------------------------

The following function translates a sentence in English to German.

.. code:: ipython3

    def translate(sentence: str) -> str:
        """
        Tokenize the sentence using the downloaded tokenizer and run the model,
        whose output is decoded into a human readable string.
    
        :param sentence: a string containing the phrase to be translated
        :return: the translated string
        """
        # Remove leading and trailing white spaces
        sentence = sentence.strip()
        assert len(sentence) > 0
        tokens = src_tokenizer.encode(sentence).ids
        # Transform the tokenized sentence into the model's input format
        tokens = [src_tokenizer.token_to_id('<s>')] + \
            tokens + [src_tokenizer.token_to_id('</s>')]
        pad_length = max_tokens - len(tokens)
    
        # If the sentence size is less than the maximum allowed tokens,
        # fill the remaining tokens with '<pad>'.
        if pad_length > 0:
            tokens = tokens + [src_tokenizer.token_to_id('<pad>')] * pad_length
        assert len(tokens) == max_tokens, "input sentence is too long"
        encoded_sentence = np.array(tokens).reshape(1, -1)
    
        # Perform inference
        enc_translated = compiled_model({input_name: encoded_sentence})
        output_key = compiled_model.output(output_name)
        enc_translated = enc_translated[output_key][0]
    
        # Decode the sentence
        sentence = tgt_tokenizer.decode(enc_translated)
    
        # Remove <pad> tokens, as well as '<s>' and '</s>' tokens which mark the
        # beginning and ending of the sentence.
        for s in ['</s>', '<s>', '<pad>']:
            sentence = sentence.replace(s, '')
    
        # Transform sentence into lower case and join words by a white space
        sentence = sentence.lower().split()
        sentence = " ".join(key for key, _ in itertools.groupby(sentence))
        return sentence

Translate the sentence 
----------------------------------------------------------------

The following function is a basic loop that translates sentences.

.. code:: ipython3

    def run_translator():
        """
        Run the translation in real time, reading the input from the user.
        This function prints the translated sentence and the time
        spent during inference.
        :return:
        """
        while True:
            input_sentence = input()
            if input_sentence == "":
                break
    
            start_time = time.perf_counter()
            translated = translate(input_sentence)
            end_time = time.perf_counter()
            print(f'Translated: {translated}')
            print(f'Time: {end_time - start_time:.2f}s')

.. code:: ipython3

    # uncomment the following line for a real time translation of your input
    # run_translator()

Test your translation 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the following cell with an English sentence to have it translated to
German

.. code:: ipython3

    sentence = "My name is openvino"
    print(f'Translated: {translate(sentence)}')


.. parsed-literal::

    Translated: mein name ist openvino.

