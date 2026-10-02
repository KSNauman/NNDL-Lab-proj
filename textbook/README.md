# KTH Textbook Index

This folder acts as the index page for the KTH action recognition project notes and experiments.

## Contents

- KTH_LRCN_Temporal_Clipping.ipynb — original temporal clipping notebook that samples frames from a KTH video and prepares them for motion-based analysis
- KTH_LRCN_Temporal_Clipping_Pipeline_Replica.ipynb — pipeline-ready continuation of the workflow with dataset preparation, frame extraction, normalization, and explanation
- KTH_LRCN_Temporal_Clipping_Updated_Replica.ipynb — updated explanatory replica of the temporal clipping workflow with clear step-by-step comments
- KTH_LRCN_Temporal_Clipping_Replica.ipynb — earlier explanatory replica of the same workflow
- LRCN_Training_Replica.ipynb — documentation of the current training-stage workflow, including dataset split, DataLoader, and LRCN model definition
- LRCN_Training_Documentation.ipynb — cleaned documentation version of the full LRCN training workflow, including evaluation metrics and interpretation
- KTH_LRCN_preprocess_attempt.ipynb — preprocessing notebook explaining the frame sampling and normalization process
- Project notes and summaries related to the LRCN pipeline
- Experimental references for preparing KTH video data before model training

## Purpose

This folder keeps the project work organized in one place. It helps track the early preprocessing steps, training ideas, and explanations related to the LRCN implementation.

## About the temporal clipping file

The file KTH_LRCN_Temporal_Clipping.ipynb focuses on temporal sampling from the KTH dataset. It selects a fixed number of frames using a constant time gap, reads each frame from the video, and prepares a compact sequence that represents motion over time.

This is important because LRCN models learn from temporal patterns instead of single static images. By extracting a sequence of frames, the model can understand action progression such as walking, running, handclapping, or handwaving.

## Index

### Preprocessing
- Temporal frame sampling
- Video frame extraction
- Resize and normalization
- Clip preparation for model input

### Model workflow
- KTH dataset loading
- Action class handling
- LRCN input pipeline
- Training preparation

## Summary

This folder is the main index for the written and experimental materials used in the KTH LRCN project. It serves as a simple guide to the project’s preprocessing and modeling process, with the temporal clipping notebook as one of the core data-preparation steps.
