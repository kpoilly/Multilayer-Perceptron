# Multilayer Perceptron

**Final note: Not evaluated yet.**


## Description:
   This project focuses on implementing a Multilayer Perceptron (MLP) from scratch for binary classification of breast cancer diagnosis.
   Learning core concepts like backpropagation and gradient descent, evaluating model performance using binary cross-entropy, and exploring advanced optimization techniques like Adam optimizer and more.

   More informations about dataset [here](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names).

## Programs:
### - separate.py :
   Separates the given data in 2 part, training set and validation set.
        
   Args: 
   
         `--train_size`, size of the training set (% of the dataset).

### - train.py :
   Trains the model on the previously separated data, using the training set for training, and the validation set for validation / early-stopping.

   Args: 
   
         `--layers`, Number of hidden layers between input and output layer
         `--layersW`, Number of neurons per layer
         `--epochs`, Number of epochs
         `--lr`, Learning rate
         `--batch_size`, Batch size
         `--patience`, Number of epochs without improvement tolerated (early stopping)
         `--clear`, Delete every models saved and their data visuals
         `--optimize`, Enable Adam optimizer
        
   Example: 
    ```
        python3 train.py --epochs 5000 --batch_size 64 --patience 5 --lr 0.00125 --optimize
    ```

### - predict.py :
   Calculates the accuracy and the Binary Entropy Loss of a given model number, or compares every models saved in a graph.
    
   Args:
   
         `--compare`, activate the comparison mode, comparing every models saved in models/ and show their accuracy and loss in a single graph.
    

## Packages needed:
    numpy
    matplotlib
    sickit-learn

![image](https://github.com/user-attachments/assets/3d4220ef-4c1f-4225-9474-f52b77bf3d06)


![image](https://github.com/user-attachments/assets/a413d016-557f-4962-914e-76308567e116)


![image](https://github.com/user-attachments/assets/7cdd47b4-9baf-43f9-98ac-4dbcb0c5d145)


