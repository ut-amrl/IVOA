# IVOA
Introspective Vision for Obstacle Avoidance

### Environment Setup

Tested with cuDNN v7.6.5 and CUDA 10.2.

Setup a virtual environment:
```bash
conda create -n ivoa-env python=3.7
```

Install dependencies:
```bash
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge scikit-image
conda install -c conda-forge pillow==6.2.1
conda install -c anaconda scikit-learn
```

### Running the Code

#### Train Dataset generation
IVOA trains and runs on patches of the input image, hence the training data for IVOA consists of a set of image patches and their corresponding labels. Before training IVOA for your obstacle detection algorithm, convert your dataset to the format expected by IVOA: It is essentially the RGB images accompanied by multiple meta data files that contain information such as
the coordinates of image patches to be extracted from the full images as well the corresponding ground truth and predicted labels for those patches by the
perception model under test. Look at the `data_preprocessing` project which converts the recorded data from AirSim along with the output of the monodepth to the IVOA dataset format.

#### Train IVOA
Modify the script `failure_detection/training/train_multi_class_model.py` and set the dictionaries `train_set_dict` and `valid_set_dict` to the session numbers for your training and the validation dataset (train, test, and validation datasets will all be loaded from the same directory and the session id determines which parts of the dataset should be used for train, test, and validation)
Modify the script `failure_detection/training/run_scripts/exec_train_multi_class.bash` to set the command line arguments according to your dataset. You need to set the path to the dataset as well as the name of the training and validation set. Params to modify based on your need in `failure_detection/training/train_multi_class_model.py` include `BATCH_SIZE` and `NUM_WORKERS`.

```bash
   ./failure_detection/training/run_scripts/exec_train_multi_class.bash
```

#### Test IVOA
<!-- #### <a name="test_ivoa_sec"></a> Test IVOA -->
1. Generate Raw Output with No Post Processing:

 Modify the script `failure_detection/testing/test_multi_class_model.py` and set the dictionary `valid_set_dict` to the session numbers for your test dataset (train, test, and validation datasets will all be loaded from the same dataset directory and the session id determines which parts of the dataset should be used for train, test, and validation).
Modify the script `failure_detection/testing/run_scripts/exec_test_multi_class.bash` to set the command line arguments according to your dataset. You need to set the path to the dataset as well as the name of the test set. Params to modify based on your need in `failure_detection/testing/test_multi_class_model.py` include `BATCH_SIZE` and `NUM_WORKERS`.

```bash
    ./failure_detection/testing/run_scripts/exec_test_multi_class.bash
```

  **NOTE:** This script runs all patches through the model and saves the resultant performance
  statistics to file. However, it does NOT visualize the output of the network. It also does not run any post processing or smoothing on the output results either.


2. Visualize (Raw Output with No Post Processing):

  For best results, IVOA should be run on overlapping patches on an input image and the output patches should be stitched together and filtered to obtain the final output. However, to get a quick but less accurate output you can run the `failure_detection/testing/visualize_inference_results.py` script. It loads each image in the test dataset and runs IVOA on NON-overlapping patches. The image is colored and saved to visualize the network's prediction.

3. Generate Output with Post Processing

	Run `failure_detection/testing/run_evaluation.py`: This is the same as `test_multi_class_model.py` but as well as saving the raw IVOA predictions on the image patches, it also saves 4 heatmaps per input image, each representing a smooth class probability for one of the 4 classes of TP, TN, FP, FN as the output of the introspection model.

	<!-- 2. Run `failure_detection/testing/run_pp_evaluation.py`: This script loads the saved heatmaps and raw prediction in step 3.1 and generates the final output of IVOA for each patch by running a post processing procedure. -->

#### Extract and Cluster Failure Mode Embeddings
In order to extract and cluster the embeddings of different failure modes. You
should run the following scripts one after another. Each script processes the
the output of the previous one in the list.

1. Run the script `failure_detection/testing/run_scripts/exec_test_multi_class.bash` after modifying the arguments according to your
dataset as explained in the [test](#markdown-header-test-ivoa) section.

1. Modify the script `failure_detection/testing/extract_embedding.py` and set the dictionary `test_set_dict` to the session numbers for your test dataset. Run `failure_detection/testing/run_scripts/exec_extract_embedding.bash` after
modifying the command line arguments according to your dataset.

1. Run `failure_detection/testing/cluster_error_embeddings.py`. This script
clusters the embeddings. Before running the
script, set the path to the embeddings extracted by `extract_embedding.py` and
the prediction results saved by `test_multi_class_model.py`. You can also
modify the clustering method of choice and its parameters.

1. Run `failure_detection/testing/dimensionality_reduction.py`. This script performs dimensionality reduction on the clusters in a way that endeavors to retain cluster separation in lower-dimensional space.
Before running the. Before running the script, set the `clustering_res_path` to the result from `cluster_error_embeddings.py`. You can also modify the dimensionality reduction parameters.

1. Run `failure_detection/testing/visualize_error_clustering.py`. This script loads the dimensionality reduction results that were saved in the previous step and visualizes random samples from each cluster. Before running the script,
you should set the paths to the outputs of `cluster_error_embeddings.py` and `dimensionality_reduction.py`
and also the `sessions_list` at the top of `main()`.



## Citation
If you find this work useful in your research, please consider citing:
```
@inproceedings{rabiee2019ivoa,
    title={IVOA: Introspective Vision for Obstacle Avoidance},
    author={Rabiee, Sadegh and Biswas, Joydeep},
    booktitle={2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year={2019},
    pages={1230-1235},
    doi={10.1109/IROS40897.2019.8968176}
}
```
