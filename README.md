# Classification Models for Early Detection of Alzheimer's Disease

 ## Project Description
 ### Goals and Objectives
 The primary goal of this project is to explore and evaluate the performance of various CNN architectures for classifying
 Alzheimer’s Disease using a 3D structural MRI scan dataset. By conducting a comparative study, our objective was to
 identify the most effective model for detecting subtle brain abnormalities associated with AD and to understand the
 strengths and limitations of each architecture. Our work provides insights into selecting the right model for various
 medical biomarkers and emphasizes the role of deep learning networks in the advancement of early diagnoses. This
 project aims to enable timely interventions that can significantly improve patient outcomes and quality of life.

 <img width="232" alt="image" src="https://github.com/user-attachments/assets/21db1e50-8215-42fe-8112-b961fbcb8f94" />

 
 ### Our Approach
 #### Dataset Collection
 We initially obtained the dataset from the Alzheimer’s Disease Neuroimaging Initiative (ADNI), a longitudinal
 multicenter observational study. Researchers use this as the most common dataset worldwide for Alzheimer’s disease
 prediction work. The database consists of biomarker data on various biomarker modalities like MRI scans, PET scans,
 genome data, etc. The general goal of ADNI is to validate biomarkers for clinical trials of Alzheimer’s disease (AD).
 Since we chose the structural MRI modality, we explored the variations available in the database. Magnetic Resonance
 Imaging (MRI) systems are classified based on the strength of their magnetic field, measured in Tesla (T). The 1.5T MRI scanner has been chosen for the dataset due to several advantages that make it a standard for clinical and research
 applications. Given the large size of the dataset and the computational requirements for preprocessing it, our project uses
 a pre-processed version of a subset of the ADNI dataset available on Kaggle. This dataset is composed of the complete
 ADNIscreening 1.5T dataset, containing preprocessed 3D MRI brain scans of 982 subjects from three groups: 221 with
 Alzheimer’s Disease (AD), 284 classified as Cognitively Normal (CN), and 477 with Mild Cognitive Impairment (MCI).
 The raw 3D MRI images in this dataset were subject to preprocessing using the FastSurfer pipeline, a fast and reliable
 tool for surface-based segmentation and cortical reconstruction. FastSurfer provides automated segmentation of the
 brain’s cortical and subcortical structures, facilitating the analysis of 3D MRI data. Detailed preprocessing steps include
 intensity normalization, skull stripping to remove non-brain elements, and segmentation of brain regions of interest.
 
 #### Dataset Preprocessing
 The data preprocessing step that we followed was done to prepare the dataset to be suitable inputs to our CNN models.
 The steps included:
 1. Masking Relevant Brain Regions: Specific brain regions are isolated based on their segmentation labels. A
 mask is created by identifying voxels that match specified label values (e.g., [17, 53, 2, 7, 41, 46] representing
 regions like hippocampus or gray matter). The mask is applied to the original brain image, retaining only
 the relevant brain regions. As a result, we obtain a new masked brain image highlighting only the regions of
 interest.
 2. Enhancing Image Contrast: CLAHE (Contrast Limited Adaptive Histogram Equalization), which improves
 local contrast and enhances details while avoiding over-amplification of noise, is used on 2D MRI slices to
 obtain enhanced 3D volumes with improved visibility of structures.
 3. Sharpening Images: Edges and fine details are highlighted by subtracting a blurred version of the image from
 the original. Parameters like radius (controls blurring) and amount (strength of sharpening) are altered. This
 gives sharpened 3D images, which improves the differentiation of features.
 4. Registration (Alignment of Images): Symmetric Diffeomorphic Registration (SDR) with the Cross-Correlation
 Metric (CCMetric) technique is used to produce a warped image aligned to the fixed reference for consistent
 anatomical positioning across subjects.
 5. Data Augmentation: The augment function applies random rotations to 3D images to increase variability and
 generalization of the training set.
 6. Resizing and Normalization: The images are resized to a fixed target shape (e.g., (100, 100, 100)) using
 anti-aliasing to ensure uniform input dimensions. After that, the images are intensity-scaled to fit the model’s
 requirements, ensuring pixel values are standardized.
 7. Dataset Organization: Images are categorized by labels (AD, MCI, CN) and organized into training and test
 sets. Additional balancing ensures equal representation of classes using oversampling or class weights. We
 use a split of 80-20 divisions into training and testing datasets, ensuring reproducibility.
 8. Batch Generation: Batches of preprocessed images and their corresponding labels are dynamically loaded.
    
 #### Models and Architectures
 
##### ADD-Net 3D
 The ADD-Net architecture consists of an input layer that takes a tensor of shape (1, 100, 100, 100). It con
sists of four convolutional blocks, each containing a 3D convolutional layer with a kernel size of (3x3x3), followed by
 a ReLU activation function and a 3D average pooling layer for dimensionality reduction. The convolutional layers
 progressively extract low-level to high-level features by increasing the number of filters across blocks. Following the
 convolutional blocks, the output is passed through a flatten layer to convert the 3D feature maps into a 1D vector. This
 is followed by two fully connected (dense) layers. The first dense layer contains 128 units with ReLU activation, while
 the second dense layer has 64 units, also with ReLU activation. Dropout layers (dropout rate = 0.5) are interspersed
 to prevent overfitting by randomly deactivating nodes during training. The final classification layer uses a SoftMax
 activation function to classify images into one of three categories corresponding to different Alzheimer’s stages. The
 model uses Adam optimizer with a learning rate of 0.001 and is trained with a batch size of 8 over one epoch. ADD-Net
 offers a superior balance of efficiency and performance because it can be designed with relatively fewer parameters
 when working with smaller datasets.
 
 ##### VGG-3D
 The VGG architecture begins with an input layer that accepts a tensor of shape (1, 100, 100, 100). It con
sists of four blocks of 3D convolutional layers with ReLU activation, progressively increasing the input and output
 feature counts (8, 16, 32, 64). The first two blocks contain two convolutional layers each, while the last two have three
 convolutional layers each. Each block is followed by a max pooling layer for dimensionality reduction and feature
 refinement. After the convolutional blocks, the data is passed through a fully connected layer with 128 units and ReLU
 activation, followed by batch normalization and dropout (dropout rate = 0.8) for regularization. A second dense layer
 with 64 units and ReLU activation is then connected to the final output layer, which uses a SoftMax activation function
 for classification into three classes. The final classification layer uses Adam optimizer with a learning rate of 0.001. It
 runs for 22 epochs, and the batch size is set to 20. VGG works well for less complex tasks and with a larger dataset.
 Therefore, pre-training it on larger datasets like ImageNet and then fine-tuning it for the given domain can improve the
 model’s efficiency. However, it needs really high computational power, which makes it a less popular choice with
 complex image classification tasks.
 
 ##### ResNet-3D
 The ResNet-3D model is a deep learning architecture designed to handle 3D data such as volumetric images
 or spatiotemporal datasets. The model builds on the traditional ResNet framework by incorporating shortcut connections
 that help the network learn residual mappings. This makes it possible to train deeper networks without encountering
 issues like vanishing gradients. To improve its ability to focus on critical features, the model includes a Convolutional
 Block Attention Module (CBAM), which uses both channel and spatial attention mechanisms to highlight the most
 important aspects of the data. The architecture starts with a 7x7x7 convolutional layer for feature extraction, followed
 by a max-pooling layer and multiple residual blocks. The number of these blocks varies depending on the chosen depth
 of the network, such as ResNet-18, ResNet-34, or ResNet-50. Each residual block contains convolutional layers with
 batch normalization and ReLU activations, while the CBAM module refines the feature representations within the
 blocks. The model ends with global average pooling and a fully connected layer, using softmax or sigmoid activation
 for the final classification. The model runs for 20 epochs with a learning rate of 0.0001 and batch size of 10.
 
 ##### 3D ResNet-18
 For this project, we used the 3D ResNet-18 model, a specialized version of the ResNet architecture designed
 to handle 3D data, such as volumetric images and spatiotemporal datasets. The ResNet-18 model relies on residual
 blocks with shortcut connections to learn residual mappings, which help address the vanishing gradient problem. These
 shortcuts allow the network to train efficiently, even as its depth increases, ensuring better performance and faster  by batch normalization, a ReLU activation, and a max-pooling layer to reduce the spatial dimensions. At its core,
 the model features four stages of residual blocks. Each stage doubles the number of feature channels and reduces
 spatial dimensions through strided convolutions. Within each block are two 3D convolutional layers, each paired with
 batch normalization and ReLU activation. When required, a downsampling operation is included to align the input and
 output dimensions. The model concludes with a global adaptive average pooling layer, which compresses the spatial
 dimensions to 1x1x1 and a fully connected layer for classification. The final classification is achieved using a Softmax
 activation function to predict across the target classes. The model runs for 20 epochs with a learning rate of 0.001 and a
 batch size of 20. The ResNet-18 model’s ability to extract spatial and temporal features from 3D data made it a good
 choice. Its efficient design ensures scalability and reliable performance, aligning well with the demands of a complex
 image classification task.
 
 ##### MobileNetV2
 The 3D MobileNetV2 neural network is an efficient, lightweight model that processes volumetric or spa
tiotemporal data, such as 3D medical scans or video segments. It extends the original MobileNetV2 architecture
 to three dimensions, utilizing Inverted Residual Blocks to maintain high accuracy while minimizing computational
 overhead. The input channels are expanded using a 1x1x1 convolution, then a 3D depthwise convolution is applied to
 capture spatial features, and finally, the result is projected back to a lower-dimensional space. This “inverted” structure
 is combined with ReLU6 activations and batch normalization. The network starts with an initial 3D convolutional
 layer and continues through a series of carefully configured Inverted Residual Blocks. The final stage produces class
 predictions using a 1x1x1 convolution and global average pooling. This model runs for 25 epochs with a learning rate
 of 0.001 and a batch size of 20.

 ## Results and Evaluation Metrics
 Weevaluated the performance of various CNN models using key metrics like Precision, Recall, F1 Score, and Accuracy.
 Wealso compared their training and testing accuracy, as shown in 3. Among the five models, VGG-3D performed the
 worst across all metrics, with a precision of 11%, recall of 29%, F1 score of 16%, and an accuracy of 32%. Its training
 accuracy is 51.95%, and its testing accuracy is 32.81%. Hence, this emphasizes its inability to extract meaningful
 features or classify the data effectively when it is complex. On the other hand, MobileNetV2 displayed a more balanced
 performance, achieving a precision of 52.8%, recall of 56%, F1 score of 54.3%, and accuracy of 53.06%. However, its
 high training accuracy of 93.33% compared to its testing accuracy of 53.06% indicates that its performance during
 training was overfitting to the train data. ADD-Net stood out with the highest testing accuracy at 69.73%, with a
 precision of 59.25% and an F1 score of 44.66%. However, its recall of 35.80% indicates the model was more focused
 on precision, which came at the cost of missing some positive instances.ResNet-3D was the top-performing model
 overall, with the highest F1 score of 75%, balanced precision and recall of 72% and 79%, and an accuracy of 75%.
 Its training accuracy of 79.07% and testing accuracy of 75% highlight its ability to generalize well while effectively
 capturing spatial and temporal features. Meanwhile, ResNet-18 achieved moderate results, with a precision of 53%,
 recall of 55%, F1 score of 54%, and accuracy of 51.02%. Despite its impressive training accuracy of 97.67%, there is a
 drop to 51.02% in testing accuracy.
 In summary, ResNet3D stood out as the best-performing model, offering both balanced and robust results. ADD-Net
 showed strong accuracy but struggled to identify all positive instances. MobileNetV2 and ResNet-18 delivered similar
 results, while VGG-3D lagged far behind in all aspects. These findings, along with the observed trends in training and
 testing accuracy, offer valuable insights into the strengths and limitations of each model and, therefore, provide a better
 basis for model selection for different tasks. 

 <img width="284" alt="image" src="https://github.com/user-attachments/assets/3f144264-c523-442b-9298-936ba1eb7458" />

 <img width="441" alt="image" src="https://github.com/user-attachments/assets/fcc443a1-f26b-4646-a65e-685a98d0d0c4" />

