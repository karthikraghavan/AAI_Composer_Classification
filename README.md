# **Classical Music Composer Classification Using Deep Learning**
### Mohammad Alhabli, Nathan Doss, Karthik Vishwanath Raghavan

---

## **Abstract**

This project leverages deep learning techniques to predict the composer of classical musical scores, focusing on four renowned composers: Bach, Beethoven, Chopin, and Mozart. The dataset, consisting of MIDI files of compositions from these composers, is collected from Kaggle, where each file is weakly labeled based on its folder or filename. The objective is to develop a model capable of classifying these compositions accurately by extracting meaningful musical features and training a deep learning model.
The methodology is structured into several phases:
Data Collection: The dataset is provided, containing MIDI files of musical compositions from Bach, Beethoven, Chopin, and Mozart. The files are labeled based on the folder or filename using a function that infers the composer from these names.


Data Preprocessing: The raw MIDI files are converted into a suitable format for deep learning. This involves parsing the MIDI data to extract symbolic music information and applying data augmentation techniques like pitch shifting and velocity jitter to diversify the dataset and improve model generalization.


Feature Extraction: Relevant features are extracted from the MIDI files using the PrettyMIDI library. These include pitch, note duration, velocity, tempo, note density, and other musical characteristics that serve as numerical representations of each composition. These features provide a compact yet informative representation of the music for the deep learning model.


Model Building: The deep learning model utilizes Long Short-Term Memory (LSTM),  Convolutional Neural Networks (CNN) and Convolutional + Recurrent Neural Networks (CRNN) to capture temporal dependencies and hierarchical musical patterns, respectively. The model is designed to classify the composer of a musical score based on the extracted features.


Model Training: The model is trained using a split of 80% training and 20% testing data, with class weights adjusted to handle the imbalance in composer distribution. The training process involves early stopping to prevent overfitting and using the Adam optimizer to minimize the loss function.


Model Evaluation: The performance of the model is evaluated using accuracy, precision, recall, and the confusion matrix. These metrics provide insight into how well the model performs across the different composers, and the results are visualized through performance curves and confusion matrices.


Model Optimization: Hyperparameter tuning, including adjustments to the network architecture and learning rate, is conducted to further improve model performance. Data augmentation techniques are fine-tuned, and additional features such as note count and tempo range are incorporated to improve classification accuracy.


The final model is expected to accurately classify musical compositions into one of four composers based on extracted features, offering an efficient tool for music analysis, especially for novice musicians and music enthusiasts. By combining LSTM and CNN architectures, the project demonstrates the effectiveness of deep learning techniques in music classification, specifically in the domain of classical music.
---

## **Overview**
Our dataset consists of curated MIDI files split into training, validation, and test sets. Each file is parsed into a fixed-size pianoroll, normalized, and label-encoded. Augmentation techniques such as pitch shifting and note dropout are applied to increase data diversity. The models are implemented using TensorFlow/Keras, with training monitored through accuracy and loss metrics. Evaluation includes classification reports, confusion matrices, and test-set predictions.

---

## **Methodology**
1. **Data Collection** – Gathering MIDI files from verified sources for Bach, Beethoven, Chopin, and Mozart.  
2. **Preprocessing** – Converting MIDI to pianoroll format, normalizing values, and label encoding.  
3. **Data Augmentation** – Pitch shifting, note dropout, and tempo variation to expand the dataset.  
4. **Model Architectures** – Implementing three models:  
   - **CNN** for spatial feature extraction from pianoroll images.  
   - **CRNN** combining convolutional and recurrent layers for temporal-spatial features.  
   - **LSTM** for sequential temporal modeling.  
5. **Training** – Using early stopping, dropout layers, batch normalization, and class weighting to improve generalization.  
6. **Model Optimization** – Hyperparameter tuning, learning rate scheduling, and architecture adjustments to improve accuracy and reduce overfitting.  
7. **Evaluation** – Accuracy, precision, recall, F1-score, and confusion matrices for each model.

---

## **Tools Used**
- **Programming Languages:** Python 3.11  
- **Deep Learning Framework:** TensorFlow/Keras  
- **Data Handling:** NumPy, Pandas, Scikit-learn  
- **MIDI Processing:** Music21, pretty_midi  
- **Visualization:** Matplotlib, Seaborn  
- **Development Environment:** Jupyter Notebook  

## **How to Run**
1. **Clone the repository**  
   ```bash
   git clone https://github.com/karthikraghavan/AAI_Composer_Classification/tree/main
   cd AAI_Composer_Classification
   ```
2. **Run the notebook**  
   Open `AAI511ComposerClassification.ipynb` in Jupyter Notebook or JupyterLab and execute the cells sequentially. 

---

## **References**
- TensorFlow/Keras Documentation  
- Music21 Documentation  
- NumPy / Pandas / Scikit-learn Documentation  
- Matplotlib / Seaborn Documentation  
- Relevant literature on symbolic music classification  

---

## **License**
Copyright (c) 2025 Mohammad Alhabli, Nathan Doss, Karthik Vishwanath Raghavan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Classical Music Composer Classification Using Deep Learning"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
