# Fall Prediction Algorithm Documentation

## Overview

Implementation of a fall prediction system using pose time series data from the [Multiple Camera Fall (MCF) dataset](https://www.iro.umontreal.ca/~labimage/Dataset/#:~:text=Multiple%20cameras%20fall%20dataset&text=This%20dataset%20contain%2024%20scenarios,ones%20contain%20only%20confounding%20events.) from University of Montreal.

## Data Collection and Processing

### Pose Detection

- Utilizing MediaPipe Pose framework to detect 33 key body landmarks in real-time
- Feature extraction includes 45 biomechanical measurements for comprehensive movement analysis

### Biomechanical Features

1. Angular Positions

   - Tracking 9 distinct body segments for complete postural analysis
   - Measuring relative angles between connected segments

2. Centroid Tracking

   - Upper body center of mass
   - Lower body center of mass
   - Total body center of mass position

3. Additional Measurements
   - Yaw trunk angle for rotational movement detection
   - 2D coordinate tracking of critical joints (hip and shoulder landmarks)

## Processing Pipeline

### Frame Processing

1. Video Frame Extraction

   - Separate extraction for activity and fall events
   - Consistent frame rate processing

2. Pose Detection

   - MediaPipe Pose implementation
   - Real-time landmark detection and tracking

3. Feature Calculation
   - Biomechanical feature computation
   - Implementation of 3-second sliding window
   - 0.5-second stride length (15 frames)
   - Binary classification labeling (fall/non-fall)

## Core Assumptions

### Anthropometric Data

- Implementation based on Plagenhoef (1983) anthropometric measurements
- Standardized body segment mass calculations

### Anatomical References

- Frontal plane definition using chest and pelvis reference points
- Body segmentation at waist level for upper/lower body analysis

## Model Architecture

### Component Layout

1. Graph Convolutional Network (GCN)

   - Input: Raw pose skeleton data
   - Output: Joint relationship features

2. Bidirectional LSTM Layer

   - Temporal pattern recognition
   - Sequence analysis

3. Feature Fusion Layer

   - Integration of GCN features with biomechanical measurements
   - Learning joint representation of spatial-temporal patterns

4. Output Layer
   - Fall probability prediction per timestep
   - Current prediction horizon: 2 seconds
   - Future goal: Extended prediction timeframe

## Class Imbalance Management

### Loss Function Implementation

1. Weighted Loss Approach

   - Enhanced weighting for fall instances
   - Dynamic weight adjustment based on class distribution

2. Performance Metrics
   - Area Under Precision-Recall Curve (AUPRC)
   - Precision measurement
   - Recall calculation
   - F1-score evaluation

## Project Deliverables

### Documentation

- Comprehensive model performance analysis
- Comparative study against baseline methodologies

### Code Repository

- GitHub repository containing:
  - Implementation code
  - Documentation
  - Usage examples
  - Testing procedures
