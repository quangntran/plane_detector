# The 3 main notebooks

1. Data generation (`generate_data.ipynb`): Get 85 images as a training set. From this training set, in each image, I extract the bounding boxes along with the label (1 for static planes and 2 for moving planes). I crop randomly 50x50 crops, check if they overlap with any of the positive (labeled 1 or 2) boxes. If they don't overlap, then the crop is accepted and its label is 0 (no planes).

    * It would be better to also label the random crops as 0 if the overlap is not significant and below a certain threshold (e.g., <30% in area of overlap is still negative). This is because in a later stage, a region proposal is not always a clear cut between having planes or not having planes, and we want our training data set to be as representative as possible to what the model sees in evaluation time.

2. Training (`train_cnn.ipynb`): Train a pre-trained VGG model on the created data set for 3 epochs (pressed for time, I could not train for longer.)

3. Predicting (`evaluation.ipynb`): Given an image, we can detect planes within each by using function `detect_img` at the very end of the notebook. This function is a process of detecting and classifying planes that can be broken down into the following stages:

    * Generating region proposals (function `get_proposals`). This uses the selective search method.
    * Filtering image using morphological means (function `get_mask`): Eliminating regions in the image that is most likely not containing any planes. `image_filtering.ipynb` also gives a visual demonstration of this process.
    * Reducing the number of proposals (function `reduce_proposals`): discard any proposals that are not within the retained region from the above step.
    * Classifying rach proposals (function `classify_proposals`): Predict the class of each of the retained proposals using the trained CNN.
    * Using maximum suppression algorithm to condense several bounding boxes for one plane into one best-fit bounding box (Not yet implemented).

# Other notebooks:
1. `show_planes.ipynb`: shows preliminary analysis on data.
2. `image_filtering.ipynb`: gives a visual demonstration of the process of filtering image using morphological means. This process is relevant and used in step 3. Predicting listed above.
