

<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="./new-ctype2.png" alt="Logo" height="180">
  </a>

  <h1 align="center">Cancer Type Classification</h1>

  <p align="center">
    Built and tested two machine learning models to classify the cancer type given a patient's miRNA profile
    <br />
    <a href="https://github.com/owscork/cancer-classification"><strong>Explore the docs »</strong></a>
    <br />
    <br />
  </p>
</div>

<h2 align="center"> Classification Models [ k-nearest-neighbors + neural network] </h2>

## Built With

![Python][Python.ico]
![Pandas][Pandas.ico]
![TF][TF.ico]
![SciKit][SciKit.ico]
![Numpy][Numpy.ico]


## About the Project
Using a large dataset containing patients' miRNA profiles that have been diagnosed with one of six cancer types, built and trained
two machine learning models to classify the type of cancer present given the miRNA profile. Utilized the k-nearest-neighbor model and 
neural network for the models, finding the neural network to produce better results.


## Data
The data provided consisted of six folders of roughly 100 patient profiles each. Folders were organized by cancer type, so all patients within a folder shared the type of cancer present, leaving a total of six different cancer types to classify from.
<br />


|miRNA_ID|read_count|reads_per_million_miRNA_mapped|cross-mapped|
| --- | --- | --- | --- |
| hsa-let-7a-1 | 108872 | 30064.300024 | N |
| hsa-let-7a-2 | 108834 | 30053.806570 | Y |
| hsa-let-7a-3 | 108895 | 30070.651326 | N |
| hsa-let-7b | 55451 | 15312.435710 | N |
| hsa-let-7a-3 | 108895 | 16407.068722 | Y |
...


[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Numpy.ico]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[SciKit.ico]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[Python.ico]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Pandas.ico]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[WebGL-url]: https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Getting_started_with_WebGL
[TF.ico]: https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white
