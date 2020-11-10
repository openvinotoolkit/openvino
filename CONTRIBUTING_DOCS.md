# Contribute to Documentation

If you want to contribute to a project documentation and make it better, your help is very welcome.
This guide puts together the guidelines to help you figure out how you can offer your feedback and contribute to the OpenVINO™ toolkit documentation.

## Contribute in Multiple ways

There are multiple ways to help improve our documentation:
* [Log an issue](https://jira.devtools.intel.com/projects/CVS/issues): Enter an issue for the OpenVINO™ documentation component for minor issues such as typos.
* Make a suggestion: Send your documentation suggestion to the mailing list.
* Contribute via GitHub: Submit your pull requests in the [GitHub](https://github.com/openvinotoolkit/openvino/tree/master/docs) documentation repository. Here the folders with the key OpenVINO documentation:
    * [https://github.com/openvinotoolkit/openvino/tree/master/docs/IE_DG](https://github.com/openvinotoolkit/openvino/tree/master/docs/IE_DG) - Inference Engine Developer Guide. Also includes Inference Engine Samples overview.
    * [https://github.com/openvinotoolkit/openvino/tree/master/docs/MO_DG](https://github.com/openvinotoolkit/openvino/tree/master/docs/MO_DG) - Model Optimizer Developer Guide.
    * [https://github.com/openvinotoolkit/openvino/tree/master/docs/get_started](https://github.com/openvinotoolkit/openvino/tree/master/docs/get_started) - Get Started Guides.
    * [https://github.com/openvinotoolkit/openvino/tree/master/docs/install_guides](https://github.com/openvinotoolkit/openvino/tree/master/docs/install_guides) - Install Guides.
    * [https://github.com/openvinotoolkit/openvino/tree/master/docs/model_server](https://github.com/openvinotoolkit/openvino/tree/master/docs/model_server) - Model Server Documentation.
    * [https://github.com/openvinotoolkit/openvino/tree/master/docs/nGraph_DG](https://github.com/openvinotoolkit/openvino/tree/master/docs/nGraph_DG) - nGraph Developer Guide.
    * [https://github.com/openvinotoolkit/openvino/tree/master/docs/ops](https://github.com/openvinotoolkit/openvino/tree/master/docs/ops) - Description of supported operations and operation sets.
    * [https://github.com/openvinotoolkit/openvino/tree/master/docs/doxygen](https://github.com/openvinotoolkit/openvino/tree/master/docs/doxygen) - Doxygen configuration files and layouts (XML files that define the documentation hierarchical structure how it's shown on [docs.openvinotoolkit.org](docs.openvinotoolkit.org)). Use the [ie_docs.xml](https://github.com/openvinotoolkit/openvino/blob/master/docs/doxygen/ie_docs.xml) file to [add](#add-new-document-to-the-documentation) or [remove](#delete-document-from-the-documentation) documents from the documentation layout. 
* Internal customers are also able to contribute to the documentation in the following OpenVINO™ closed repositories in GitHub and GitLab:
    * [https://gitlab-icv.inn.intel.com/algo/post-training-compression-tool](https://gitlab-icv.inn.intel.com/algo/post-training-compression-tool) - Post-Training Optimization Toolkit
    * [https://github.com/openvinotoolkit/workbench](https://github.com/openvinotoolkit/workbench) - Deep Learning Workbench.

## Contribute via GitHub

Use the following steps to contribute in the OpenVINO™ Toolkit documentation.

### Use Documentation Guidelines
The documentation for our project is written using Markdown. Use our [guidelines](./docs/documentation_guidelines.md) and best practices to write consistent, readable documentation:

* **[Authoring Guidelines](./docs/documentation_guidelines.md#authoring-guidelines)**
* **[Structure Guidelines](./docs/documentation_guidelines.md#structure-guidelines)**
* **[Formatting Guidelines](./docs/documentation_guidelines.md#structure-guidelines)**
* **[Graphics Guidelines](./docs/documentation_guidelines.md#graphics-guidelines)**

### Add New Document to the Documentation
> **NOTE**: Please check if that information can be added to existing documents instead of creating a new one.

1. Fork the [OpenVINO™ Toolkit](https://github.com/openvinotoolkit/openvino) repository.
2. Create a new branch.
3. Create a new markdown file in an appropriate folder.
    > **REQUIRED**: The document title must contain a document label in a form: `{#openvino_docs_<name>}`. For example: `Deep Learning Network Intermediate Representation and Operation Sets in OpenVINO™ {#openvino_docs_MO_DG_IR_and_opsets}`. This label is a unique identifier that is used to refer to the document from others.
4. Add your file to the documentation structure. Open the documentation structure file [docs/doxygen/ie_docs.xml](./docs/doxygen/ie_docs.xml) and add your file path to the appropriate section.
5. Commit changes to your branch.
6. Create a pull request, choose a reviewer or assign to yourself. 
7. Once the pull request is created, automatic checks are started. All checks must pass to continue.
8. Discuss, review, and update your contributions.
9. Get merged once the maintainer and reviewers approve.

### Edit Existing Document
1. Fork the [OpenVINO™ Toolkit](https://github.com/openvinotoolkit/openvino) repository.
2. Create a new branch.
3. Edit the documentation markdown file and commit changes to the branch.
4. Create a pull request choose a reviewer or assign to yourself.
5. Once the pull request is created, automatic checks are started. All checks must pass to continue.
6. Discuss, review, and update your contributions.
7. Get merged once the maintainer and reviewers approve.

### Delete Document from the Documentation
1. Fork the [OpenVINO™ Toolkit](https://github.com/openvinotoolkit/openvino) repository.
2. Create a new branch.
3. Remove the documentation file.
4. Remove your file from the documentation structure. Open the documentation structure file [docs/doxygen/ie_docs.xml](./docs/doxygen/ie_docs.xml) and remove all occurrences of your file path.
5. Remove all references to that file from other documents or replace with links to alternatives topics (if any).
6. Commit changes to your branch.
7. Create a pull request, choose a reviewer or assign to yourself.
8. Once the pull request is created, automatic checks are started. All checks must pass to continue.
9. Discuss, review, and update your contributions.
10. Get merged once the maintainer and reviewers approve.

## See Also
[How to Contribute Models to Open Model Zoo](https://github.com/ntyukaev/open_model_zoo/blob/master/CONTRIBUTING.md)
