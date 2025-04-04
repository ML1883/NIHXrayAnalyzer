# NIHXrayAnalyzer
Image classification model using the NIHCC chest xray dataset and Python.

It uses a relatively standard 5 layer CNN model augmented by Gaussian attention layers that are trained by using bounding box data provided with the dataset. Note that not all pictures have bounding box data and that the effect of the attention layers can be rather limited, depending on the finding that it is trying to classify (some are more predictable in terms of location than others.)

The NIH dataset can be found in this article: https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

Though I would like to create a great model, this repo is really just to try out different machine learning methods, see what works best and what doesn't, try out different data visualization, etc. As such, the contents may be very messy. And, of course, no warranties are provided at all. This means I also deliberately opted not to perform a literature review to check what is the current 'best' performer and improve on that. That is not the point. The reason for e.g. choosing the Gaussian attention layers is because it "makes sense" (at least from my non-medical perspective) for some findings to roughly occur at similar positions of the lungs.

Things that I still want to do:
* Changing positions of the attention layer
* Converting code to Jax
* ?

Python version used is 3.13. Mostly build upon Pytorch and perhaps later on Jax.