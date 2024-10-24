# TODO:

1. Implement transforms/normalizations on the data
    * Perspective and Joint rotation transforms
    * Random rotation and squeeze in 2D space
    * Duplicate end and start frames to give signs consistent length
    * Normalize coordinates based on signing space
    * Normalize hand coordinates based on hand bounding box, so that the model learns the shape of the hands
2. Test the accuracy of using 3D coordinates, specifically how accurate z-coordinate estimates are.
3. Detect sign start and end frames