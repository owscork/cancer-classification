

<div align="center">

  <h1 align="center">Classifying Cancer Type w/ Machine Learning</h1>

  <p align="center">
    Built, trained, and tested classification models to predict the cancer type for a given patient's miRNA profile
    <br />
  </p>
</div>

## Table of Contents:

- [Built with](#built-with)
- [About the Project](#about-the-project)
- [Data](#data)
  + [Sample miRNA Profile Snippet](#sample-mirna-profile-snippet)
- [Usage](#usage)


## Built With

![Python][Python.ico]
![Pandas][Pandas.ico]
![TF][TF.ico]
![SciKit][SciKit.ico]
![Numpy][Numpy.ico]


## About the Project

Utilizing the TensorFlow Keras library, two classification models were created to predict the type of cancer present given the patient's miRNA profile. The models were trained on a provided dataset consisting of hundreds of patients' miRNA profiles and the corresponding cancer type found. Six types of cancer are covered in the dataset, so the models are only capable of predicting between those six types, which is why classification models are necessary for this project's purpose.

## Data

The data provided consisted of six folders of roughly 100 patient profiles each. Folders were organized by cancer type, so all patients within a folder shared the same type. The contents of a patient profile comprised of almost 2000 miRNA markers, otherwise referred to as features, with values recorded for each representing the amount that feature was expressed in the patient's tissue.

### Sample miRNA Profile Snippet

Below is an example of a miRNA profile provided in the data:
|miRNA_ID|read_count|reads_per_million_miRNA_mapped|cross-mapped|
| --- | --- | --- | --- |
| hsa-let-7a-1 | 108872 | 30064.300024 | N |
| hsa-let-7a-2 | 108834 | 30053.806570 | Y |
| hsa-let-7a-3 | 108895 | 30070.651326 | N |
| hsa-let-7b | 55451 | 15312.435710 | N |
| hsa-let-7a-3 | 108895 | 16407.068722 | Y |
...

The miRNA_ID column indicates the names of the miRNA markers or features measured in a patient's tissue. The reads_per_million_miRNA_mapped column provides a normalized value for each feature representing the level it was expressed in the tissue. 


## Usage




[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Numpy.ico]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[SciKit.ico]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[Python.ico]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Pandas.ico]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[WebGL-url]: https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Getting_started_with_WebGL
[TF.ico]: https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white
