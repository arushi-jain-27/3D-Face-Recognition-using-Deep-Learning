# 3D-Face-Recognition-using-Deep-Learning

In this work, we develop a generic tool for the recognition of similar objects. The techniques proposed for object recognition mainly focus on categorizing heterogeneous objects. However, when subjected to the multi-class classification problem of similar objects, these models don't fare so well. PointNet architecture is one such model that directly consumes point clouds of images as inputs, instead of relying on the much bulkier 3D voxels and grids, to classify multiple objects. Even though it gives remarkable accuracy when classifying different objects, the performance declines when we try to classify similar objects like faces. 

We are proposing a solution that combines the object classification utility from PointNet architecture along with One-Shot Learning from Siamese Network that converts our multi-class classification problem to a binary classification problem and improves object recognition accuracy, even for similar objects. We are applying our proposed approach on 3D face recognition by conducting a series of experiments on three 3D face databases, namely, IIT Indore database, Bosphorus database, and University of Notre Dame (UND) database, to test our model. 

We also use a novel data augmentation technique that uses sub-sampling from the existing point clouds to increase the size and variability of the available data. 

The experimental results show that the proposed method is considerably better in recognizing objects that are highly similar as compared to the original PointNet architecture.
