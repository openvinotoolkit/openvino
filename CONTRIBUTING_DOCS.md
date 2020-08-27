# Contribute to Documentation

If you want to contribute to a project documentation and make it better, your help is very welcome.
This guide puts together the guidelines to help you figure out how you can offer your feedback and contribute to the documentation.

## Contribute in Multiple ways

There are multiple ways to help improve our documentation:
* [Log an issue](https://jira.devtools.intel.com/projects/CVS/issues): Enter an issue for the OpenVINO™ documentation component for minor issues such as typos.
* Make a suggestion: Send your documentation suggestion to the mailing list.
* Contribute via GitHub: Submit pull requests in the [GitHub](https://github.com/openvinotoolkit/openvino/tree/master/docs) documentation repository.

## Contribute via GitHub

Use the following steps to contribute in the OpenVINO™ Toolkit documentation

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
    > **REQUIRED**: The document title must contain a document label in a form: `{#openvino_docs_<name>}`. For example: `Deep Learning Network Intermediate Representation and Operation Sets in OpenVINO™ {#openvino_docs_MO_DG_IR_and_opsets}`.
4. Add your file to the documentation structure. Open the documentation structure file [docs/doxygen/ie_docs.xml](./docs/doxygen/ie_docs.xml) and add your file path to the appropriate section.
5. Commit changes to your branch.
6. Create a pull request.
7. Once the pull request is created, automatic checks are started. All checks must pass to continue.
8. Discuss, review, and update your contributions.
9. Get merged once the maintainer approves.

### Edit Existing Document
1. Fork the [OpenVINO™ Toolkit](https://github.com/openvinotoolkit/openvino) repository.
2. Create a new branch.
3. Edit the documentation markdown file and commit changes to the branch.
4. Create a pull request.
5. Once the pull request is created, automatic checks are started. All checks must pass to continue.
6. Discuss, review, and update your contributions.
7. Get merged once the maintainer approves.

### Delete Document from the Documentation
1. Fork the [OpenVINO™ Toolkit](https://github.com/openvinotoolkit/openvino) repository.
2. Create a new branch.
3. Remove the documentation file.
4. Remove your file from the documentation structure. Open the documentation structure file [docs/doxygen/ie_docs.xml](./docs/doxygen/ie_docs.xml) and remove all occurences of your file path.
5. Remove all references to that file from other documents or replace with links to alternatives topics (if any).
6. Commit changes to your branch.
7. Create a pull request.
8. Once the pull request is created, automatic checks are started. All checks must pass to continue.
9. Discuss, review, and update your contributions.
10. Get merged once the maintainer approves.
