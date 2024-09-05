# PresiSAM

PreciSAM automatically transforms imprecise bounding boxes into precise bounding boxes. PreciSAM works actually with [SAM2](https://github.com/facebookresearch/segment-anything-2) models.

It increases the quality of annotations for computer vision datasets.

You can now annotate your data faster, leaving the tedious work of annotation precision to PreciSAM.

## How does it work?

1. Importing a dataset in COCO format (dataset coming from CVAT for now)
2. Retrieving bounding box annotations from your images.
3. Passage of each of the bounding boxes to the SAM2 model
4. The main object of each bounding box is segmented by SAM2
5. Recording the bounding box surrounding the segmented object

## Installation

1. Clone the repository
```
git clone https://github.com/Youho99/PreciSAM.git
cd preciSAM
```

2. Install dependencies with:

```
pip install -r requirements
```

## Getting Started

* Launch the command:

```
streamlit run precisam.py
```

The application opens in your web browser. If it did not open automatically, copy the URL that appears in your console, and paste it into your web browser.

* Choose your SAM2 model

* Import your dataset in COCO format (CVAT, in .zip)

* Click on the button to start the process

The first time you use a SAM2 model, it is downloaded into cache. The download may take several minutes.

* When the process is complete, click the download button to retrieve your converted dataset with the accurate bounding boxes.