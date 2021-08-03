# Spectral-Normalized-Gaussian-Process

This is a PyTorch implementation of the Spectral Normalized Gaussian Process. I decided to make this since I couldn't find any PyTorch implementations of it, and thought it was pretty neat.

Why use SNGP? Deep learning models are notoriously poor at accurately returning a confidence score. A DL model will be 99.99% certain about something that it is dead wrong about. In many domains, this can be a significant problem, and it's important for the model to have some idea of when it's out of its depth (pun intended). Classically, Gaussian Processes are a fantastic ML model for computing uncertainty.

GPs are essentially big fancy function interpolaters that work under the assumption that whatever function you want to model is reasonably smooth (think lines like handwriting and NOT lines like stock prices). If you're a physicist, it's like finding the ground state of a classical field with a pairwise potential and no kinetic energy.

Anyway, this repo is heavily influenced by the TensorFlow source code for their GP layer. In some cases the comments are directly copied (where I got too lazy to write), but mostly all of the code is reworked, and I added some imo better class stuff like @property decorators.

To use this, you can use the GP layer directly "RandomFeatureGaussianProcess" or you can use the SNGP classifier layer, which just combines some standard operations. That one is good to go for applying as a classification head to some input hidden dimension (N, D) (it adds the spec norm wrappers on linear layers).

Finally, I wrote up a notebook that translates the TensorFlow tutorial, but added some additionaly stuff like an MC dropout comparison. It's a great starting point to see why SNGP is useful and how to implement it. Check it out!
