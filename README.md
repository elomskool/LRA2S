# 🌍 LRA2S
## Linear regression applied 2 sismology

Linear Regression for Seismological Subduction Analysis Off the Coast of Japan

### Overview

This project applies a linear regression model to explore and quantify relationships within seismic data collected off the Japanese coast. Specifically, it focuses on understanding the subduction of the Pacific oceanic plate beneath the Eurasian continental plate, a major tectonic process responsible for intense seismic activity in the region.

Using recorded depths of earthquakes and their horizontal distances from the subduction trench, we aim to model how seismic depth varies as a function of distance — a proxy for the angle and dynamics of subduction.

## 🎯 Objectives

Model the relationship between earthquake depth and distance from trench using linear regression
Provide an interpretable approximation of the subduction angle through the slope of the regression line.
Enable debugging and visual inspection of the model’s performance during training.

Store and visualize key training metrics such as slope (m), intercept (c), and mean squared error (MSE) over training epochs.

## 🔬 Scientific Background

The Japan Trench marks a convergent plate boundary where the Pacific Plate subducts beneath the Eurasian Plate. Earthquakes in this region vary in depth depending on their distance from the trench, reflecting the geometry of the descending slab.

By modeling this relationship with a simple linear regression, we can:
Approximate the subduction angle (via the slope).
Investigate potential anomalies or deviations from idealized tectonic behavior.

Build an educational foundation for integrating geophysical models with machine learning techniques.

## technology used and libraries necessary to run the program
### Python3.9, Numpy, Pandas, Matplotlib

pip install numpy <br />
pip install pandas <br />
pip install matplotlib <br />


