# fish_classification_with_ANN

This project aims to classify different fish species using a deep-learning model. The project uses a large-scale fish dataset downloaded from Kaggle and is implemented with an artificial neural network model built using TensorFlow/Keras.

## Project Details
 
- **Dataset**: A large-scale fish dataset from Kaggle.
- **Model**: Artificial Neural Network (ANN) built using TensorFlow/Keras.
- **Techniques**: Deep learning techniques for image classification.
- **Goal**: To accurately classify fish species based on images.

This project uses deep learning techniques to solve the problem of fish species classification. Users can perform classification operations using a trained model to identify fish species in the dataset correctly.

### Model Architecture
- **Input Layer:** 
  - The input layer is designed to handle images of shape (224, 224, 3), which are the dimensions of the fish images.
  
- **Hidden Layers:**
  - **First Hidden Layer:** 512 neurons with ReLU activation function.
  - **Second Hidden Layer:** 256 neurons with ReLU activation function.
  - **Third Hidden Layer:** 128 neurons with ReLU activation function.

### Hyperparameters
- **Optimizer:** Adam optimizer was chosen for its efficiency and ability to adapt the learning rate during training.
- **Learning Rate:** A learning rate of 0.001 was used, which provided a good balance between convergence speed and stability.
- **Batch Size:** A batch size of 32 was selected to ensure efficient use of computational resources while maintaining stable training.
- **Epochs:** The model was trained for 50 epochs, sufficient for the model to converge without overfitting.
- **Dropout:** Dropout rates of 0.3 were used in the hidden layers to prevent overfitting by randomly dropping neurons during training.

### Rationale for Hyperparameter Selection
- **Neurons and Layers:** The number of neurons and layers were chosen based on initial experiments and tuning. A higher number of neurons and layers can capture more complex patterns and increase the risk of overfitting. The chosen architecture provided a good balance between complexity and generalization.
- **Dropout:** A dropout rate of 0.3 was effective in preventing overfitting while maintaining model performance.
- **Optimizer and Learning Rate:** Adam optimizer with a learning rate of 0.001 provided efficient and stable training dynamics.

## Usage
Users can perform classification operations using the trained model to identify fish species in the dataset correctly.

## Project link
[Fish Classification with ANN on Kaggle](https://www.kaggle.com/code/remdilatkse/fish-classification-with-ann)
