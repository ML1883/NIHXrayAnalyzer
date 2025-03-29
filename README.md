# NIHXrayAnalyzer
Simple image classification model using the NIHCC chest xray dataset and Python.

It uses a relatively standard 5 layer CNN model augmented by Gaussian attention layers that are trained by using bounding box data provided with the dataset.

This mostly to try out image classification methods, different neural nets, etc.

The NIH dataset can be found in this article: https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

Though I would like to create a great model, this repo is really just to try out different machine learning methods, see what works best and what doesn't, get a hang over visualizations methods, etc. As such, the contents may be very messy. And, of course, no warranties are provided at all.

Python version used is 3.13. Mostly build upon Pytorch and perhaps later on Jax.