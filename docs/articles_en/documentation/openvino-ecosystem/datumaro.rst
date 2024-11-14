Datumaro
========


.. meta::
   :description: Start working with Datumaro, which offers functionalities for basic data
                 import/export, validation, correction, filtration and transformations.


Datumaro provides a suite of basic data import/export (IE) for more than 35 public vision data
formats and manipulation functionalities such as validation, correction, filtration, and some
transformations. To achieve the web-scale training, this further aims to merge multiple
heterogeneous datasets through comparator and merger. Datumaro is integrated into Geti™, OpenVINO™
Training Extensions, and CVAT for the ease of data preparation. Datumaro is open-sourced and
available on `GitHub <https://github.com/openvinotoolkit/datumaro>`__.
Refer to the official `documentation <https://openvinotoolkit.github.io/datumaro/stable/docs/get-started/introduction.html>`__ to learn more.
Plus, enjoy `Jupyter notebooks <https://github.com/openvinotoolkit/datumaro/tree/develop/notebooks>`__ for the real Datumaro practices.

Detailed Workflow
#################

.. image:: ../../assets/images/datumaro.png

1. To start working with Datumaro, download public datasets or prepare your own annotated dataset.

   .. note::
      Datumaro provides a CLI `datum download` for downloading `TensorFlow Datasets <https://www.tensorflow.org/datasets>`__.

2. Import data into Datumaro and manipulate the dataset for the data quality using `Validator`, `Corrector`, and `Filter`.

3. Compare two datasets and transform the label schemas (category information) before merging them.

4. Merge two datasets to a large-scale dataset.

   .. note::
      There are some choices of merger, i.e., `ExactMerger`, `IntersectMerger`, and `UnionMerger`.

5. Split the unified dataset into subsets, e.g., `train`, `valid`, and `test` through `Splitter`.

   .. note::
      We can split data with a given ratio of subsets according to both the number of samples or
      annotations. Please see `SplitTask` for the task-specific split.

6. Export the cleaned and unified dataset for follow-up workflows such as model training.
Go to :doc:`OpenVINO™ Training Extensions <openvino-training-extensions>`.

If the results are unsatisfactory, add datasets and perform the same steps, starting with dataset annotation.

Datumaro Components
###################

* `Datumaro CLIs <https://openvinotoolkit.github.io/datumaro/stable/docs/command-reference/overview.html>`__
* `Datumaro APIs <https://openvinotoolkit.github.io/datumaro/stable/docs/reference/datumaro_module.html>`__
* `Datumaro data format <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/datumaro_format.html>`__
* `Supported data formats <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/index.html>`__

Tutorials
#########

* `Basic skills <https://openvinotoolkit.github.io/datumaro/stable/docs/level-up/basic_skills/index.html>`__
* `Intermediate skills <https://openvinotoolkit.github.io/datumaro/stable/docs/level-up/intermediate_skills/index.html>`__
* `Advanced skills <https://openvinotoolkit.github.io/datumaro/stable/docs/level-up/advanced_skills/index.html>`__

Python Hands-on Examples
########################

* `Data IE <https://openvinotoolkit.github.io/datumaro/stable/docs/jupyter_notebook_examples/dataset_IO.html>`__
* `Data manipulation <https://openvinotoolkit.github.io/datumaro/stable/docs/jupyter_notebook_examples/manipulate.html>`__
* `Data exploration <https://openvinotoolkit.github.io/datumaro/stable/docs/jupyter_notebook_examples/explore.html>`__
* `Data refinement <https://openvinotoolkit.github.io/datumaro/stable/docs/jupyter_notebook_examples/refine.html>`__
* `Data transformation <https://openvinotoolkit.github.io/datumaro/stable/docs/jupyter_notebook_examples/transform.html>`__
* `Deep learning end-to-end use-cases <https://openvinotoolkit.github.io/datumaro/stable/docs/jupyter_notebook_examples/e2e_example.html>`__



