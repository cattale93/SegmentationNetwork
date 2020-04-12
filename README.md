# SegmentationNetwork
Implementation of a Deep Convolutional Neural Network Full Patch Labelling. Land cover classification.

#### Scripts
- Data_downsampling.ipynb is jupyter notebook file and it is used to downsample images to 1x1 meter resolution
- Data_processing_2_0.ipynb is used to fuse data and cut tiles
- Split_Train_Test_Val_FINALE.ipynb is used to split data in train, test and validation
- mapper.ipynb is used to create the map of a whole tile

#### Classes and Functions
- cut it is a function used to split tiles in patches, it took as input the percentage of acceptable dead pixels and the percentage of overlap between patches
- DataGenerator_ImageSeg this class is used to dynamically load data to train the network. It returns a batch of GT and a batch of data applying data augmentation if requested. It works indexing all data to do not overload memory.

#### Software and Hardware
All the algorithm has been developed using Google Colaboratory. It provides a good computational power for free for limited time. It supports also Graphic Processing Unit (GPU), the model is not specified but could be a Nvidia K80s, T4s, P4s and P100s none knows. Practically, network cannot be trained using this platform because of the limitation of time and and resources, these GPU are not the latest fashion. The training processes has been run using dedicated machine of the Remote Sensing Laboratory (RSlab) of University of Trento (UniTN) with GPU Nvidia RTX 2070 super, a Central Processing Unit (CPU) intel i7-9800x and 64GB of Random Access Memory (RAM).
