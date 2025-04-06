# NIHXrayAnalyzer
Image classification model using the NIHCC chest xray dataset and Python.

The model consists of the following parts:
1. 4 layers of CNN
2. Gaussian attention layers using bounding box data which is aviable for 1k of the >100k pictures
3. Standard neural nets to classify 

The best performance seems to be achieved by a high dropout penalty in part 3 of the model. I suspect that if you want to have production ready models you are probably best of making a specialized model that is trained on a diagnoses-by-diagnoses sample. Then you could probably also tell the CNN in case of smaller diagnoses like masses and such to look more specifically for that diagnosis. This could be enhanced by more bounding box data to come to a great model. I say this because the evaluation of the models constantly shows that the 'larger' (i.e. takes up more space visually) a diagnosis is, the better the AUC. You can see this in the main notebook where I made some ROC curves. This of course also has some implications of the practical usability of the model (the 'larger' a diagnosis is, the more obvious it is to spot).   

The NIH dataset can be found in this article: https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

Though I would like to create a great model, this repo is really just to try out different machine learning methods, see what works best and what doesn't, try out different data visualization, etc. As such, the contents may be very messy. And, of course, no warranties are provided at all. This means I also deliberately opted not to perform a literature review to check what is the current 'best' performer and improve on that. That is not the point. The reason for e.g. choosing the Gaussian attention layers is because it "makes sense" (at least from my non-medical perspective) for some findings to roughly occur at similar positions of the lungs.

Things that I still want to do:
* Changing positions of the attention layer
* Converting code to Jax
* ?

Python version used is 3.13. Mostly build upon Pytorch and perhaps later on Jax.