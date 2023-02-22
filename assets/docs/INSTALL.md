# Installation

1. Clone and enter the repository:
```
git clone git@github.com:AIS-Bonn/OCVP-object-centric-video-prediction.git
cd OCVP-object-centric-video-prediction
```


2. Install all required packages by installing the ```conda``` environment file included in the repository:
```
conda env create -f environment.yml
conda activate OCVP
```


3. Download the Obj3D and MOVi-A datasets, and place them under the `datasets` directory. The folder structure should be like:
```
OCVP
├── datasets/
|   ├── Obj3D/
|   └── MOViA/
```

 * **Obj3D:** Donwload and extract this dataset by running the following bash script:
 ```
 chmod +x download_obj3d.sh
 ./download_obj3d.sh
 ```

 - **MOViA:** Download the MOVi-A dataset to your local disk from the [Google Cloud Storage](https://console.cloud.google.com/storage/browser/kubric-public/tfds), and  preprocess the *TFRecord* files to extract the video frames and other required metadata by running the following commands:
 ```
 gsutil -m cp -r gs://kubric-public/tfds/movi_a/128x128/ .
 mkdir movi_a
 mv 128x128/ movi_a/128x128/
 python src/extract_movi_dataset.py
 ```



4. Download and extract the pretrained models, including checkpoints for the SAVi decomposition and prediction modules:
```
chmod +x download_pretrained.sh
./download_pretrained.sh
```
