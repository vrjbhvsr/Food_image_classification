
# Food Image Classification using Deep Learning





## Table of content

* Demo
* Overview
* Dataset
* Methodology
* Installation
* Tech
* License
* Credit
* Contributing
* Authors
## Demo

![ezgif com-crop](https://github.com/vrjbhvsr/Food_image_classification/assets/158619347/b2daacdf-9367-4b90-8218-fb38acdb3ef1)


## Overview

This repository contains code for a Food Image Classification project aimed at developing a machine learning model capable of accurately classifying images of food into 20 distinct categories. The project utilizes deep learning techniques and convolutional neural networks (CNNs) to classify food images, with the goal of providing valuable insights for various applications such as dietary tracking, menu recommendation systems, and food recognition for visually impaired individuals.
## Dataset

The dataset that has been used is Food-101. The dataset is available at kaggle website.
The Linik for the dataset is: https://www.kaggle.com/datasets/dansbecker/food-101.

The Food-101 dataset is consist of 101 different classes and 1000 images which i have reduce to 20 classes and 400 images. The modification code is provided as `data_modification.py`.
## Methodology
**1. Data Ingestion** : Zipfile of dataset from the google drive will be downloaded in the artifacts folder and process of unzipping will start. The dataset already separated in train and test dataset.

**2. Data Transformation** : Image data will be transformed with the help of `torchvision.transforms`. For example, data will be resized, cropped, rotated, fliped,etc.. Those transformed will be created into dataset and then converted into dataloaders.

**3. Base Model** : A pretrained Deep Convolutional Network, ResNet101, is chosen to extract relevant features from the input images. With the help of transfer learning techniques such as, unfreezing the 4th layer while keeping parameters as it is for rest of the layers can be very helpful to improve model performance.

**4. Model Training** : The model is trained using a portion of the dataset, with the loss function optimized using techniques such as Adam optimization. Hyperparameter tuning and regularization techniques are employed to prevent overfitting and improve model performance.

**5. Evaluation** : The trained model's performance is evaluated using a separate test set, measuring metrics such as accuracy. Model performance is analyzed across different food categories to identify areas for improvement.

**6. Model Pushing** : Model pushing has done with the help of `Bentoml` service. BentoMl service is created and the it saved. This will serialize model, dependencies, and configuration into a format that can be easily loaded and deployed. 

## Installation


**1. Clone the repository:**

```bash
  git clone https://github.com/vrjbhvsr/Food_image_classification.git
```

**2. Install Dependencies**
```bash
  pip install -r requirements.txt
```

**3.** Run the scripts in the src/ directory to preprocess data, train models, and evaluate performance. 

**4.** Run the `app.py` to start the Flask Web server, and the application should be accessible at specified address (usually http://localhost:5000/ by default).
## Tech Stack

![download](https://github.com/vrjbhvsr/Food_image_classification/assets/158619347/ef296e45-34d6-4fde-b700-65e0168561c6)


![download (2)](https://github.com/vrjbhvsr/Food_image_classification/assets/158619347/aec41bf7-1c2f-421f-83ce-7333fb039862)

![download (1)](https://github.com/vrjbhvsr/Food_image_classification/assets/158619347/8b8c95a1-c944-4aef-af9a-9ab8b27a4a03)
![download (3)](https://github.com/vrjbhvsr/Food_image_classification/assets/158619347/759ff0fd-0717-44c5-b680-c071317cbe00)


## License

This project is licensed under the MIT License. See the LICENSE file for details.
## Credit

* [kaggel Dataset](https://www.kaggle.com/datasets/dansbecker/food-101)
* [InuronTeam](https://ineuron.ai/)

## Contributing

Contributions to improve the project are welcome! If you find any bugs or have suggestions for enhancements, please open an issue or submit a pull request.
## Authors

- [@VrajBhavsar](https://github.com/vrjbhvsr)

