# Deep-Convolutional-Network-for-Thorax-Disease-Detection

	Using PyTorch, pre-processed the Chest X-ray dataset that contains 112,120 X-ray images multi-labeled with 14 diseases from 30,805 unique patients by performing z-score on per image basis and data augmentation by downsizing to 244×244, random rotation, center crop, and random horizontal ﬂip.

	Self-designed a 9-layer CNN for training, and a 6-layer ResNet for ensembling.

	Used Xavier weight initialization and batch normalization to accelerate training process, adopted dropout to regularize and utilized Weighted Binary Cross Entropy Loss to address the unbalanced multi-class problem and got accuracy of 89.03% and BCR of 17.57%.

	Tried a pre-trained DenseNet-169 and fine-tuned the model.
