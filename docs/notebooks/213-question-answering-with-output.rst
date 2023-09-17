Interactive question answering with OpenVINO™
=============================================

This demo shows interactive question answering with OpenVINO, using
`small BERT-large-like
model <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/bert-small-uncased-whole-word-masking-squad-int8-0002>`__
distilled and quantized to ``INT8`` on SQuAD v1.1 training set from
larger BERT-large model. The model comes from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo/>`__. Final part
of this notebook provides live inference results from your inputs. 

**Table of contents:**

- `Imports <#imports>`__ 
- `The model <#the-model>`__ 

  - `Download the model <#download-the-model>`__ 
  - `Load the model <#load-the-model>`__ 
  
    - `Select inference device <#select-inference-device>`__ 

- `Processing <#processing>`__ 

  - `Preprocessing <#preprocessing>`__ 
  - `Postprocessing <#postprocessing>`__ 
  - `Main Processing Function <#main-processing-function>`__ 
  
- `Run <#Run>`__

  - `Run on local paragraphs <#run-on-local-paragraphs>`__ 
  - `Run on websites <#run-on-websites>`__

Imports
###############################################################################################################################

.. code:: ipython3

    import operator
    import time
    from urllib import parse
    
    import numpy as np
    from openvino.runtime import Core
    
    import html_reader as reader
    import tokens_bert as tokens

The model
###############################################################################################################################

Download the model
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Use ``omz_downloader``, which is a command-line tool from the
``openvino-dev`` package. The ``omz_downloader`` tool automatically
creates a directory structure and downloads the selected model. If the
model is already downloaded, this step is skipped.

You can download and use any of the following models:
``bert-large-uncased-whole-word-masking-squad-0001``,
``bert-large-uncased-whole-word-masking-squad-int8-0001``,
``bert-small-uncased-whole-word-masking-squad-0001``,
``bert-small-uncased-whole-word-masking-squad-0002``,
``bert-small-uncased-whole-word-masking-squad-int8-0002``, just change
the model name in the code below. All of these models are already
converted to OpenVINO Intermediate Representation (OpenVINO IR), so
there is no need to use ``omz_converter``.

.. code:: ipython3

    # A directory where the model will be downloaded.
    base_model_dir = "model"
    
    # Selected precision (FP32, FP16, FP16-INT8).
    precision = "FP16-INT8"
    
    # The name of the model from Open Model Zoo.
    model_name = "bert-small-uncased-whole-word-masking-squad-int8-0002"
    
    model_path = f"model/intel/{model_name}/{precision}/{model_name}.xml"
    model_weights_path = f"model/intel/{model_name}/{precision}/{model_name}.bin"
    
    download_command = f"omz_downloader " \
                       f"--name {model_name} " \
                       f"--precision {precision} " \
                       f"--output_dir {base_model_dir} " \
                       f"--cache_dir {base_model_dir}"
    ! $download_command


.. parsed-literal::

    ################|| Downloading bert-small-uncased-whole-word-masking-squad-int8-0002 ||################
    
    ========== Downloading model/intel/bert-small-uncased-whole-word-masking-squad-int8-0002/vocab.txt
    
    
    ========== Downloading model/intel/bert-small-uncased-whole-word-masking-squad-int8-0002/FP16-INT8/bert-small-uncased-whole-word-masking-squad-int8-0002.xml
    
    
    ========== Downloading model/intel/bert-small-uncased-whole-word-masking-squad-int8-0002/FP16-INT8/bert-small-uncased-whole-word-masking-squad-int8-0002.bin
    
    


Load the model
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Downloaded models are located in a fixed structure, which indicates a
vendor, a model name and a precision. Only a few lines of code are
required to run the model. First, create an OpenVINO Runtime object.
Then, read the network architecture and model weights from the ``.xml``
and ``.bin`` files. Finally, compile the network for the desired device.
You can choose ``CPU`` or ``GPU`` for this model.

.. code:: ipython3

    # Initialize OpenVINO Runtime.
    core = Core()
    # Read the network and corresponding weights from a file.
    model = core.read_model(model_path)

Select inference device
-------------------------------------------------------------------------------------------------------------------------------

Select device from dropdown list for running inference using OpenVINO:

.. code:: ipython3

    import ipywidgets as widgets
    
    core = Core()
    
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

    compiled_model = core.compile_model(model=model, device_name=device.value)
    
    # Get input and output names of nodes.
    input_keys = list(compiled_model.inputs)
    output_keys = list(compiled_model.outputs)
    
    # Get the network input size.
    input_size = compiled_model.input(0).shape[1]

Input keys are the names of the input nodes and output keys contain
names of output nodes of the network. There are 4 inputs and 2 outputs
for BERT-large-like model.

.. code:: ipython3

    [i.any_name for i in input_keys], [o.any_name for o in output_keys]




.. parsed-literal::

    (['input_ids', 'attention_mask', 'token_type_ids', 'position_ids'],
     ['output_s', 'output_e'])



Processing
###############################################################################################################################

NLP models usually take a list of tokens as a standard input. A token is
a single word converted to some integer. To provide the proper input,
you need the vocabulary for such mapping. You also need to define some
special tokens, such as separators or padding and a function to load the
content from provided URLs.

.. code:: ipython3

    # The path to the vocabulary file.
    vocab_file_path = "../data/text/bert-uncased/vocab.txt"
    
    # Create a dictionary with words and their indices.
    vocab = tokens.load_vocab_file(vocab_file_path)
    
    # Define special tokens.
    cls_token = vocab["[CLS]"]
    pad_token = vocab["[PAD]"]
    sep_token = vocab["[SEP]"]
    
    
    # A function to load text from given urls.
    def load_context(sources):
        input_urls = []
        paragraphs = []
        for source in sources:
            result = parse.urlparse(source)
            if all([result.scheme, result.netloc]):
                input_urls.append(source)
            else:
                paragraphs.append(source)
    
        paragraphs.extend(reader.get_paragraphs(input_urls))
        # Produce one big context string.
        return "\n".join(paragraphs)

Preprocessing
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The input size in this case is 384 tokens long. The main input
(``input_ids``) to used BERT model consists of two parts: question
tokens and context tokens separated by some special tokens.

If ``question + context`` are shorter than 384 tokens, padding tokens
are added. If ``question + context`` is longer than 384 tokens, the
context must be split into parts and the question with different parts
of context must be fed to the network many times.

Use overlapping, so neighbor parts of the context are overlapped by half
size of the context part (if the context part equals 300 tokens,
neighbor context parts overlap with 150 tokens). You also need to
provide the following sequences of integer values:

-  ``attention_mask`` - a sequence of integer values representing the
   mask of valid values in the input.
-  ``token_type_ids`` - a sequence of integer values representing the
   segmentation of ``input_ids`` into question and context.
-  ``position_ids`` - a sequence of integer values from 0 to 383
   representing the position index for each input token.

For more information, refer to the **Input** section of `BERT model
documentation <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/bert-small-uncased-whole-word-masking-squad-int8-0002#input>`__.

.. code:: ipython3

    # A generator of a sequence of inputs.
    def prepare_input(question_tokens, context_tokens):
        # A length of question in tokens.
        question_len = len(question_tokens)
        # The context part size.
        context_len = input_size - question_len - 3
    
        if context_len < 16:
            raise RuntimeError("Question is too long in comparison to input size. No space for context")
    
        # Take parts of the context with overlapping by 0.5.
        for start in range(0, max(1, len(context_tokens) - context_len), context_len // 2):
            # A part of the context.
            part_context_tokens = context_tokens[start:start + context_len]
            # The input: a question and the context separated by special tokens.
            input_ids = [cls_token] + question_tokens + [sep_token] + part_context_tokens + [sep_token]
            # 1 for any index if there is no padding token, 0 otherwise.
            attention_mask = [1] * len(input_ids)
            # 0 for question tokens, 1 for context part.
            token_type_ids = [0] * (question_len + 2) + [1] * (len(part_context_tokens) + 1)
    
            # Add padding at the end.
            (input_ids, attention_mask, token_type_ids), pad_number = pad(input_ids=input_ids,
                                                                          attention_mask=attention_mask,
                                                                          token_type_ids=token_type_ids)
    
            # Create an input to feed the model.
            input_dict = {
                "input_ids": np.array([input_ids], dtype=np.int32),
                "attention_mask": np.array([attention_mask], dtype=np.int32),
                "token_type_ids": np.array([token_type_ids], dtype=np.int32),
            }
    
            # Some models require additional position_ids.
            if "position_ids" in [i_key.any_name for i_key in input_keys]:
                position_ids = np.arange(len(input_ids))
                input_dict["position_ids"] = np.array([position_ids], dtype=np.int32)
    
            yield input_dict, pad_number, start
    
    
    # A function to add padding.
    def pad(input_ids, attention_mask, token_type_ids):
        # How many padding tokens.
        diff_input_size = input_size - len(input_ids)
    
        if diff_input_size > 0:
            # Add padding to all the inputs.
            input_ids = input_ids + [pad_token] * diff_input_size
            attention_mask = attention_mask + [0] * diff_input_size
            token_type_ids = token_type_ids + [0] * diff_input_size
    
        return (input_ids, attention_mask, token_type_ids), diff_input_size

Postprocessing
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The results from the network are raw (logits). Use the softmax function
to get the probability distribution. Then, find the best answer in the
current part of the context (the highest score) and return the score and
the context range for the answer.

.. code:: ipython3

    # Based on https://github.com/openvinotoolkit/open_model_zoo/blob/bf03f505a650bafe8da03d2747a8b55c5cb2ef16/demos/common/python/openvino/model_zoo/model_api/models/bert.py#L163
    def postprocess(output_start, output_end, question_tokens, context_tokens_start_end, padding, start_idx):
    
        def get_score(logits):
            out = np.exp(logits)
            return out / out.sum(axis=-1)
    
        # Get start-end scores for the context.
        score_start = get_score(output_start)
        score_end = get_score(output_end)
    
        # An index of the first context token in a tensor.
        context_start_idx = len(question_tokens) + 2
        # An index of the last+1 context token in a tensor.
        context_end_idx = input_size - padding - 1
    
        # Find product of all start-end combinations to find the best one.
        max_score, max_start, max_end = find_best_answer_window(start_score=score_start,
                                                                end_score=score_end,
                                                                context_start_idx=context_start_idx,
                                                                context_end_idx=context_end_idx)
    
        # Convert to context text start-end index.
        max_start = context_tokens_start_end[max_start + start_idx][0]
        max_end = context_tokens_start_end[max_end + start_idx][1]
    
        return max_score, max_start, max_end
    
    
    # Based on https://github.com/openvinotoolkit/open_model_zoo/blob/bf03f505a650bafe8da03d2747a8b55c5cb2ef16/demos/common/python/openvino/model_zoo/model_api/models/bert.py#L188
    def find_best_answer_window(start_score, end_score, context_start_idx, context_end_idx):
        context_len = context_end_idx - context_start_idx
        score_mat = np.matmul(
            start_score[context_start_idx:context_end_idx].reshape((context_len, 1)),
            end_score[context_start_idx:context_end_idx].reshape((1, context_len)),
        )
        # Reset candidates with end before start.
        score_mat = np.triu(score_mat)
        # Reset long candidates (>16 words).
        score_mat = np.tril(score_mat, 16)
        # Find the best start-end pair.
        max_s, max_e = divmod(score_mat.flatten().argmax(), score_mat.shape[1])
        max_score = score_mat[max_s, max_e]
    
        return max_score, max_s, max_e

First, create a list of tokens from the context and the question. Then,
find the best answer by trying different parts of the context. The best
answer should come with the highest score.

.. code:: ipython3

    def get_best_answer(question, context):
        # Convert the context string to tokens.
        context_tokens, context_tokens_start_end = tokens.text_to_tokens(text=context.lower(),
                                                                         vocab=vocab)
        # Convert the question string to tokens.
        question_tokens, _ = tokens.text_to_tokens(text=question.lower(), vocab=vocab)
    
        results = []
        # Iterate through different parts of the context.
        for network_input, padding, start_idx in prepare_input(question_tokens=question_tokens,
                                                               context_tokens=context_tokens):
            # Get output layers.
            output_start_key = compiled_model.output("output_s")
            output_end_key = compiled_model.output("output_e")
    
            # OpenVINO inference.
            result = compiled_model(network_input)
            # Postprocess the result, getting the score and context range for the answer.
            score_start_end = postprocess(output_start=result[output_start_key][0],
                                          output_end=result[output_end_key][0],
                                          question_tokens=question_tokens,
                                          context_tokens_start_end=context_tokens_start_end,
                                          padding=padding,
                                          start_idx=start_idx)
            results.append(score_start_end)
    
        # Find the highest score.
        answer = max(results, key=operator.itemgetter(0))
        # Return the part of the context, which is already an answer.
        return context[answer[1]:answer[2]], answer[0]

Main Processing Function
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Run question answering on a specific knowledge base (websites) and
iterate through the questions.

.. code:: ipython3

    def run_question_answering(sources, example_question=None):
        print(f"Context: {sources}", flush=True)
        context = load_context(sources)
    
        if len(context) == 0:
            print("Error: Empty context or outside paragraphs")
            return
    
        if example_question is not None:
            start_time = time.perf_counter()
            answer, score = get_best_answer(question=example_question, context=context)
            end_time = time.perf_counter()
    
            print(f"Question: {example_question}")
            print(f"Answer: {answer}")
            print(f"Score: {score:.2f}")
            print(f"Time: {end_time - start_time:.2f}s")
        else:
            while True:
                question = input()
                # if no question - break
                if question == "":
                    break
    
                # measure processing time
                start_time = time.perf_counter()
                answer, score = get_best_answer(question=question, context=context)
                end_time = time.perf_counter()
    
                print(f"Question: {question}")
                print(f"Answer: {answer}")
                print(f"Score: {score:.2f}")
                print(f"Time: {end_time - start_time:.2f}s")

Run
###############################################################################################################################

Run on local paragraphs
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Change sources to your own to answer your questions. You can use as many
sources as you want. Usually, you need to wait a few seconds for the
answer, but the longer the context, the longer the waiting time. The
model is very limited and sensitive for the input. The answer can depend
on whether there is a question mark at the end. The model will try to
answer any of your questions even if there is no good answer in the
context. Therefore, in such cases, you can see random results.

Sample source: a paragraph from `Computational complexity
theory <https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/Computational_complexity_theory.html>`__

Sample questions:

-  What is the term for a task that generally lends itself to being
   solved by a computer?
-  By what main attribute are computational problems classified
   utilizing computational complexity theory?
-  What branch of theoretical computer science deals with broadly
   classifying computational problems by difficulty and class of
   relationship?

If you want to stop the processing just put an empty string.

**First, run the code below. If you want to run it in interactive mode
set ``example_question`` as ``None``, run the code, and then put your
questions in the box.**

.. code:: ipython3

    sources = ["Computational complexity theory is a branch of the theory of computation in theoretical computer "
               "science that focuses on classifying computational problems according to their inherent difficulty, "
               "and relating those classes to each other. A computational problem is understood to be a task that "
               "is in principle amenable to being solved by a computer, which is equivalent to stating that the "
               "problem may be solved by mechanical application of mathematical steps, such as an algorithm."]
    
    question = "What is the term for a task that generally lends itself to being solved by a computer?"
    
    run_question_answering(sources, example_question=question)


.. parsed-literal::

    Context: ['Computational complexity theory is a branch of the theory of computation in theoretical computer science that focuses on classifying computational problems according to their inherent difficulty, and relating those classes to each other. A computational problem is understood to be a task that is in principle amenable to being solved by a computer, which is equivalent to stating that the problem may be solved by mechanical application of mathematical steps, such as an algorithm.']
    Question: What is the term for a task that generally lends itself to being solved by a computer?
    Answer: A computational problem
    Score: 0.51
    Time: 0.03s


Run on websites
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

You can also provide URLs. Note that the context (a knowledge base) is
built from paragraphs on websites. If some information is outside the
paragraphs, the algorithm will not be able to find it.

Sample source: `OpenVINO
wiki <https://en.wikipedia.org/wiki/OpenVINO>`__

Sample questions:

-  What does OpenVINO mean?
-  What is the license for OpenVINO?
-  Where can you deploy OpenVINO code?

If you want to stop the processing just put an empty string.

**First, run the code below. If you want to run it in interactive mode
set ``example_question`` as ``None``, run the code, and then put your
questions in the box.**

.. code:: ipython3

    sources = ["https://en.wikipedia.org/wiki/OpenVINO"]
    
    question = "What does OpenVINO mean?"
    
    run_question_answering(sources, example_question=question)


.. parsed-literal::

    Context: ['https://en.wikipedia.org/wiki/OpenVINO']
    Question: What does OpenVINO mean?
    Answer: Open Visual Inference and Neural network Optimization
    Score: 0.94
    Time: 0.06s

