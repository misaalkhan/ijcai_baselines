# NutriAI: AI-Powered Child Malnutrition Assessment in Low-Resource Environments

Welcome to the NutriAI GitHub repository! This project, developed by a team from the Indian Institute of Technology Jodhpur and the All India Institute of Medical Science Jodhpur, seeks to tackle the global issue of child malnutrition by employing AI technology. The paper discussing this project has been accepted in the International Joint Conference on Artificial Intelligence (IJCAI) 2023 under the AI for Social Good Track.

## Introduction
Malnutrition is a significant public health issue, especially in children under the age of five. Despite multiple interventions by various governments and organizations, the issue persists, contributing to a significant proportion of morbidity and mortality in this age group. NutriAI aims to address this challenge using AI to efficiently and cost-effectively monitor and assess children's nutritional status.

## Dataset
We are in the process of collecting data and have successfully assembled three different types of datasets:

1. **MalDB**: Internet-curated image samples taken in uncontrolled settings, annotated by professional doctors. MalDB contains 3,090 images, divided equally between the healthy and unhealthy classes. The images are further categorized by skin tone.

2. **IITJ (Adult)**: Collected images of adult students at IITJ for transfer learning in children. The dataset includes all anthropometric measurements and binary labels using BMI. There are a total of 479 images, 80 subjects, and six different poses along with the following anthropometric measurements: Height, Weight, MUAC, HC, Waist Circumference, Age, and Gender.

3. **AIIMSJ and Govt. School (Child Health Dataset)**: Collected images of children from clinical and community health settings. It contains all anthropometric measurements and binary labels based on WHO and CDC standards. The dataset contains a total of 2729 images corresponding to 334 subjects, 6 different poses, and the following anthropometric measurements: Height, Weight, MUAC, HC, Waist Circumference, Age, and Gender.

We also used the full body image and BMI dataset of celebrities for transfer learning to supplement the small sample size of the IITJ dataset. You can find this dataset [here](https://github.com/atoms18/BMI-prediction-from-Human-Photograph).

Please note, the datasets are still being collected and will only be made available upon completion, with proper permissions and upon request. This repository will be updated accordingly.

## Baseline Models
The baseline models provided in this repository are trained on the datasets mentioned above. They can be used for testing once the datasets are made available.

### Architectures Used

We evaluated several widely recognized deep learning architectures:

- **ResNet:** 
  - ResNet18
  - ResNet50
  - ResNet101

- **VGG:**
  - VGG16
  - VGG19

- **DenseNet121**


The models were fine-tuned by adding three additional convolutional layers, each followed by ReLU activation, batch normalization, and dropout. The models were trained using the Adam optimizer, with a learning rate of 0.001, over 10 epochs, with a batch size of 32. 

To ensure fair evaluation, we employed a train-test-validation split of 60-20-20.

## Dataset Augmentation
The dataset was also augmented by applying transformations that do not alter the shape of individuals in the image.

## Results
The results demonstrated the model's performance on the task of binary classification, i.e., healthy and unhealthy. 

| Dataset              | ResNet18 | ResNet50 | ResNet101 | VGG16 | VGG19 | DenseNet |
|----------------------|----------|----------|-----------|-------|-------|----------|
| MalDB                | 70.65    | 71.34    | 71.65     | 69.40 | 71.03 | 70.97    |
| IITJ (Adult)         | 63.10    | 63.03    | 62.30     | 64.20 | 64.70 | 63.19    |
| Child Health Dataset | 63.20    | 67.34    | 63.66     | 68.5  | 67.78 | 65.50    |


Note that the dataset will continue to be updated and expanded.

# Trained Models

The following models have been trained on each of the three datasets. 

## MalDB Dataset

| Model | Link |
| ----- | ---- |
| ResNet18 | [Download](<https://drive.google.com/file/d/1wcLDmsc9toARgDks0d1ndL4y5LwCxpKn/view?usp=share_link>) |
| ResNet50 | [Download](<https://drive.google.com/file/d/1mgVt_VKao2H1VPWPvFwuobGMzuwV-jm_/view?usp=share_link>) |
| ResNet101 | [Download](<https://drive.google.com/file/d/16hOwNJ-Y2uN2-gsveGi5DMKzn-7B2Vry/view?usp=share_link>) |
| VGG16 | [Download](<https://drive.google.com/file/d/1B-FDG4Dv7FfR6DL-VpJthO37Drak80Ed/view?usp=share_link>) |
| VGG19 | [Download](<https://drive.google.com/file/d/1TauUwjMh8lmuvAVQ0ic9rj1V4zwiILuP/view?usp=share_link>) |
| DenseNet | [Download](<https://drive.google.com/file/d/1A-aHuwEE6lnP_H4l8txKi7cBeRCeXobg/view?usp=share_link>) |

## IITJ (Adult) Dataset

| Model | Link |
| ----- | ---- |
| ResNet18 | [Download](<https://drive.google.com/file/d/1FJPT8VpCbHrCtoT2SzYnnJbpYmmQ5rJT/view?usp=share_link>) |
| ResNet50 | [Download](<https://drive.google.com/file/d/1Dx0WP37UkNH3nKulDbvpfMDneuI2Esq2/view?usp=share_link>) |
| ResNet101 | [Download](<https://drive.google.com/file/d/1woJgejsatEdsbGlQ7b_TNlOJorN18i0G/view?usp=share_link>) |
| VGG16 | [Download](<https://drive.google.com/file/d/1ad4uPg19QA7fn1rdxaCm9_HRWg22eZdm/view?usp=share_link>) |
| VGG19 | [Download](<https://drive.google.com/file/d/1Qg6LPrYJRmCVMvrqkHFEh9k1VJkww-XJ/view?usp=share_link>) |
| DenseNet | [Download](<https://drive.google.com/file/d/1XwvFa19rpWjIHQy62JP8ydrs-zMIIr32/view?usp=share_link>) |

## Child Health Dataset

| Model | Link |
| ----- | ---- |
| ResNet18 | [Download](<https://drive.google.com/file/d/151eOYkF7e4PL0sCYTS5RPHlNc5eTZqH8/view?usp=share_link>) |
| ResNet50 | [Download](<https://drive.google.com/file/d/1k6JEpGXIDql-KEVdOJzgQsVi028N5sP2/view?usp=share_link>) |
| ResNet101 | [Download](<https://drive.google.com/file/d/1u4FDKHDuPY3ZjIwmN4vhxHXruPclbBfP/view?usp=share_link>) |
| VGG16 | [Download](<https://drive.google.com/file/d/1lMGVR35RT2LeOh9E9zZ2iwQGT8bURNeq/view?usp=share_link>) |
| VGG19 | [Download](<https://drive.google.com/file/d/13RCO_Fjnb_Tgksi0i3m1Zxy6gGOrZfWQ/view?usp=share_link>) |
| DenseNet | [Download](<https://drive.google.com/file/d/1S0ecBp2L-yGx69ZIEkhay_5cvBlKUbSK/view?usp=share_link>) |


## How to Use the Code
Once you've downloaded the models and datasets, you can run the code by adding the appropriate paths in the script.

## Authors
This project was developed by:

- Misaal Khan, Indian Institute of Technology Jodhpur, India and All India Institute of Medical Science Jodhpur, India
- Shivang Agarwal, Indian Institute of Technology Jodhpur, India
- Mayank Vatsa, Indian Institute of Technology Jodhpur, India
- Richa Singh, Indian Institute of Technology Jodhpur, India
- Kuldeep Singh, All India Institute of Medical Science Jodhpur, India

## Citation
If you use the code or the baseline models in this repository, please consider citing our paper:

```bibtex
@inproceedings{khan2023nutriai,
  title={NutriAI: AI-Powered Child Malnutrition Assessment in Low-Resource Environments},
  author={Khan, Misaal and Agarwal, Shivang and Vatsa, Mayank and Singh, Richa and Singh, Kuldeep},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2023}
}
```

## Acknowledgements
We would like to express our gratitude to the healthcare professionals who contributed their expertise in classifying the samples. Their valuable input has been crucial in the development and validation of these models.

## Contact
For any queries or further information about the project, please feel free to get in touch with us.

## License
This project is licensed under the terms of the MIT license.