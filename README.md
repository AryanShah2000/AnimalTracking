# AnimalTracking: Deep Learning for Animal Track Classification

This project applies deep learning techniques to automatically classify animal tracks from images. Using both Convolutional Neural Networks (ResNet-18) and Vision Transformers (ViT), we trained models to identify 18 different animal species using the Open Animal Tracks dataset. The project includes data preprocessing, model training and evaluation, and an interactive web app for real-time predictions.

## Project Files

| File Name                                 | Description |
|------------------------------------------|-------------|
| `OpenAnimalTracks-ResNet-Standard.ipynb` | Trains a ResNet-18 model with all layers unfrozen. The notebook includes standard preprocessing, model training, evaluation, and result visualization. |
| `OpenAnimalTracks-ResNet-WithFreezing.ipynb` | Implements a training strategy where early layers of the ResNet-18 model are frozen. Only the classification head and deeper layers are fine-tuned. |
| `OpenAnimalTracks-Transformer.ipynb`     | Uses a Vision Transformer (ViT) to classify animal tracks. Includes training with both full fine-tuning and progressive unfreezing strategies. |
| `animal_tracks_model20.pth`              | A saved PyTorch model file (likely after 20 epochs of training). It contains the trained weights and can be reloaded for inference or further training. |
| `Deep Learning Blog.pdf`                 | A comprehensive final report describing the full project lifecycle: data preprocessing, model selection, training details, evaluation results, and a demonstration app. Includes visualizations and lessons learned. |

## Models Used

- **ResNet-18**: A convolutional neural network architecture that uses skip connections to improve training efficiency and depth. Fine-tuned for the 18-class classification task.
- **Vision Transformer (ViT)**: A transformer-based model that splits images into patches and applies self-attention to model relationships across the image. Both standard and progressive unfreezing methods were used during training.

## Results

- Top performing species included Mule Deer (87% accuracy), Goose (83%), and Turkey (78%).
- The model had difficulty distinguishing between similar species like Gray Fox and Bobcat, with only 28% classification accuracy.
- ResNet-18 achieved about 65% overall accuracy.
- The Vision Transformer model showed early signs of overfitting, reaching 60% accuracy with standard fine-tuning and 58% with progressive unfreezing.

## Web Application

We developed a Streamlit-based web application that allows users to upload an image of an animal footprint. The app predicts the species using our trained ResNet-18 model and provides contextual information such as habitat and fun facts. The goal is to make the technology accessible for researchers, conservationists, and enthusiasts in the field.

## Future Work

- Expand the dataset with more images and a broader range of species
- Include images from multiple perspectives to improve spatial understanding
- Improve model robustness through enhanced data augmentation and regularization
- Develop a mobile version of the application for easier field use

## Contributors

Aryan Shah, Brinda Asuri, Haden Loveridge, Sam Chen, Sarah Dominguez, Kimble Horsack

## References

- Open Animal Tracks Dataset
- PyTorch and torchvision documentation
- Vision Transformer models via the `timm` library
- Streamlit for web application development

This project was developed as part of a deep learning course and demonstrates the practical application of machine learning to ecological research.
