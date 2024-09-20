# Historical Text Recognition using CRNN Model

This project aims to address the challenge of text recognition from `historical Spanish printed sources` dating back to the `seventeenth century`, a domain where existing Optical Character Recognition (OCR) tools often fail due to the complexity and variability of the texts. Leveraging hybrid end-to-end models based on a combination of CNN and RNN architectures, namely `CNN-RNN`, our research seeks to develop advanced machine learning techniques capable of accurately transcribing non-standard printed text. This project is a part of the `RenAIssance project`, a large project under the HumanAI organization. I am `Shashank Shekhar Singh`, a third year student from `IIT BHU, India` and have been developing this project as a part of the `Google Summer of Code program' 2024`.

<p align="center">
  <img src="images/humanai_logo.jpg" alt="HumanAI" style="height: 100px; margin-right: 20px;"/>
  <img src="images/gsoc_logo.png" alt="GSOC" style="height: 50px; padding-bottom: 50px" />
</p>

## Table of Contents

- [Project Goals](#Project-Goals)
- [Installation](#installation)
- [About The Project](#About-The-Project)
- [Datasets and Models](#datasets-and-models)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Links](#links)

## Project Goals

1. **Development of Hybrid End-to-End Models:** The primary goal of this project is to design, implement, and fine-tune hybrid end-to-end models based on CRNN architectures for text recognition. By combining the strengths of architectures such as recurrent neural networks (RNN) and convolutional neural networks (CNN), the models aim to effectively capture both local and global features in the historical Spanish printed text enhancing accuracy and robustness in transcription.
2. **Achieving High Accuracy:** The ultimate objective was to train machine
learning models capable of extracting text from seventeenth-century Spanish printed sources with at least **80%** accuracy. This entails extensive experimentation, hyperparameter tuning, and dataset curation to ensure the models generalize well across various styles, fonts, and degradation levels present in historical documents. Achieving this goal will signify a significant advancement in text recognition, particularly in the context of preserving and analyzing ancient textual artifacts.

## Installation

You don't need to install anything externally, just fire up the python notebook on your favourite coding platform (Google Colab, Jupter Notebook, Kaggle etc) and start running the code cells one after the other. All the packages that need to be installed are kept as the first code block in the Python Notebook.

### Project Directory Structure
1. **Dataset_Generation.ipynb** - It is a Python Notebook to generate training data from book PDF and Transcription. If you just want to train and test the CRNN model, you can directly skip running this notebook.

2. **Model.ipynb** - It is a standalone Python Notebook that is used for model training and inferencing. It is trained on a corrected and modified data generated during the course of the GSoC period.

## About The Project

#### Irregularities and Ambiguities
- **Interchangeable Characters**: Characters like 'u' & 'v', and 'f' & 's' were used interchangeably. Assume 'u' at the beginning of word and 'v' inside word. Assume 's' at the beginning/end of a word, 'f' within a word.
- **Tildes (horizontal “cap” – ignore grave/backwards accents)**: 
    1. When a q is capped, assume ue follows
    2. When a vowel is capped, assume n follows
    3. When n is capped, this is always the letter ñ
- **Old Spellings**: ç old spelling is always modern z.
- **Line End Hyphens**: Some line end hyphens not present. Leaving words split for now.

#### Dataset and Pre-processing
- **Input Data:** The main dataset consists of 31 scanned pages: 25 have transcriptions available, the last 6 pages of transcriptions have been removed to later evaluate the degree of accuracy and viability of the test method employed.
- **PDF and DOC to Images Folder**: This flowchart depicts the path followed to generate the dataset for training the CRNN Model.
    <p align="center"><img src="images/Pre_Process.png" alt="CRNN Architecture" style="height: 500px; margin-right: 20px;"/><p>
- **CRAFT Model**: The CRAFT model for Bounding Box Detection and Localisation gives the following results.
    <p align="center">
    <img src="images/imageOriginal.png" alt="Before CRAFT Model" style="height: 400px; margin-right: 20px;"/>
    <img src="images/imageCRAFT.jpg" alt="After CRAFT Model" style="height: 400px;" />
    </p>
- **Enhancements**: Augmentation techniques like rotation and Gaussian noise addition.

#### Model Architecture

- **CRNN Model**: The Convolutional Recurrent Neural Networks is the combination of two of
the most prominent neural networks. The CRNN (convolutional recurrent
neural network) involves CNN (convolutional neural network) followed by
the RNN (Recurrent neural networks).
<p align="center"><img src="images/CRNN.png" alt="CRNN Architecture" style="height: 400px; margin-right: 20px;"/></p>

- ***CNN***: CNNs are used for extracting spatial features from input images, transforming them into a feature map.
<p align="center"><img src="images/CNN.png" alt="CNN Architecture" style="height: 300px; margin-right: 20px;"/></p>

- ***RNN***: RNNs then process these sequentially to capture contextual dependencies and predict character sequences.
<p align="center"><img src="images/RNN.png" alt="RNN Architecture" style="height: 300px; margin-right: 20px;"/></p>

- ***Current CRNN Architecture***: The model plot represents the CRNN architecture that has been trained in the Python Notebook shared above.
<p align="center"><img src="images/CRNN_Plot.png" alt="CRNN Architecture" style="height: 600px; margin-right: 20px;"/></p>

#### Training and Evaluation
- **Hyperparameter Optimization**: Selection through vast amount of experimentation.
- **Model Calibration**: Utilizes validation loss and other techniques to align sequence likelihoods with quality, improving output accuracy.
- **Evaluation Metrics**: Performance evaluated using CTC Loss and Validation loss.

- ***Loss vc Epochs***: The model has been made quite performant and light weight. It get's an optimum amount of training in just 10-15 epochs.
<p align="center"><img src="images/Loss.png" alt="Learning Curve" style="height: 300px; margin-right: 20px;"/></p>

For a detailed walkthrough of the project's development, challenges, and solutions, read the complete blog post [here](https://medium.com/@shashankshekharsingh1205/my-journey-with-humanai-in-the-google-summer-of-code24-program-part-2-bb42abce3495).

## Datasets and Models
- The `Padilla - Nobleza virtuosa_testExtract.pdf` can be downloaded from [here](https://github.com/Shashankss1205/RenAIssance/blob/main/RenAIssance_CRNN_OCR_Shashank_Shekhar_Singh/data/Padilla_Nobleza_virtuosa_testExtract.pdf) 
- The `Padilla - 1 Nobleza virtuosa_testTranscription.docx` can be downloaded from [here](https://github.com/Shashankss1205/RenAIssance/blob/main/RenAIssance_CRNN_OCR_Shashank_Shekhar_Singh/data/Padilla_Nobleza_virtuosa_testTranscription.docx) 
- The ocr model used can be directly generated by running the python notebook or can be downloaded from [here](https://github.com/Shashankss1205/RenAIssance/blob/main/RenAIssance_CRNN_OCR_Shashank_Shekhar_Singh/Model/ocr_model.h5)

## Model Performance

| Metric | Value |
|--------|-------|
| Character Accuracy | 95.79% |
| CER | 0.027 |
| CTC Loss | 0.1 |
| Validation Loss | 0.07 |

## Acknowledgements

This project is supported by the [HumanAI Foundation](https://humanai.foundation/) and Google Summer of Code 2024. Detailed documentation and a journey of this project can be found on my [blog post](https://medium.com/@shashankshekharsingh1205/my-journey-with-humanai-in-the-google-summer-of-code24-program-part-2-bb42abce3495).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Links

- [Google Summer of Code 2024 Project](https://summerofcode.withgoogle.com/programs/2024/projects/lg7vQeMM)
- [HumanAI Foundation](https://humanai.foundation/)

Feel free to fork the repository and submit pull requests. For major changes, please open an issue to discuss your ideas first. Contributions are always welcomed!