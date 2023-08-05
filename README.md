# Reverse Image Search (Content Based Image Retrieval) using low-dimensional embeddings

## Objective

Given an image, find other similar images from a learned set of images.

## Notebook

See [CBIR.ipynb](https://github.com/huyndao/content-based-image-retrieval/blob/main/CBIR.ipynb) for the implementation.

### Use Cases

-   Find similar images (or duplicates)
-   Find source image (e.g. which directory / link it originated from)
    regardless of if the image has the same name or not
-   If image has multiple versions at different resolutions, can find all of
    them
-   Copyright violation detection

### Advantages

-   No human metadata labeling required
-   Analyze images based on shapes / tones / textures w/in each image

### Disadvantages

-   Slow, especially on large image set
-   Requires some hyperparameter tuning
-   Likely does not work well on images that have been flipped or rotated

### Examples of Existing Implementations

-   Google Image Search
-   TinEye
-   [image-match](https://github.com/rhsimplex/image-match) (Opensource but no longer maintained)
-   [ImageHash](https://github.com/JohannesBuchner/imagehash) (Opensource, has multiple image hashing algorithms)

### Why roll my own implementation?

-   To see if I could
-   Privacy: allows me to search similar images from my private albums, instead
    of uploading to some web service by Google or TinEye
-   The advantages of both `image-match` and `ImageHash` are that they are much
    faster than this implementation due to simpler hashing / signature
    calculations
-   This implementation maybe more accurate than the above opensource
    implementation since the low-dimensional embeddings were tuned specifically
    for each dataset, i.e. data points (images) more similar to each other are
    moved closer and points more different are pushed further apart in the
    low-Dimensional manifold
-   Additionally, the former implementations were designed specifically for
    images, while this approach can be re-tooled to be applied to any data type
    (e.g. audio or texts) as long as they can be converted to numbers

## Implementation Overview

A `CBIR` class is built to the following framework:

1.  Walk the path, seach and store all image names
2.  Build an image array with `PIL` by processing each image as followed:

    -   Convert to greyscale
    -   Resize to 64x64
    -   Reshape to a 1-Dimensional array (i.e. shape 1x4096)

3.  Fit UMAP (Uniform Manifold Approximation & Projection) algorithm with the
    following hyperparemeters

    -   w/ specified `n_neighs` neighbors (this is to be tuned)
    -   w/ specified `n_comps` components (i.e. dimensions) (select an
        appropriate number based on speed and compute power availability)
    -   `min_d` = 0.0 for `min_dist` (pre-set so that similar points clump
        together)

4.  Tune the `n_neighs` hyperparameter by calculating the `trustworthiness`
    score

    -   The trustworthiness score ranges from 0 to 1, with the higher end
        indicating a better lower dimensional representation of the dataset
    -   Plot trustworthiness score versus k neighbors
    -   Select an appropriate k neighbors
    -   [Trustworthiness
        Score](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html)

5.  Plug the `n_neighs` = `k` neighbors from the previous step to the final model
6.  Save final model
7.  Query an image against the built dataset as desired

