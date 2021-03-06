# pitip

PITIP is a TEM image processor (Under construction)

## Contour

Detect the contour of the aimed object and colormap the curvature of each point on the contour.  

Thanks to [morphsnakes](https://github.com/pmneila/morphsnakes), the contour is well dectected using *Morphological Snakes* [1][2] method. After smoothing the contour with an average convolution, we calculate the [curvature](https://en.wikipedia.org/wiki/Curvature) of the contour by fitting a circle to a point and the two points that are several points away from it[3]. 

A Demo:  
<img src="/images/futaba_rio.gif" height="295">
<img src="/images/futaba_rio_cv.png" height="295">

Note that active contour detection can be slow on high resolution pictures, 
one may compress the picture to accelerate the detection. Superpixel algorithm might be useful in such case (TODO).

## Particle analysis

Particle detection and distance analysis.

A Demo:  
<img src="/images/particles_analysis.png" height="295">
<img src="/images/particles_distance.png" height="295">

## References

[1]: *A Morphological Approach to Curvature-based Evolution of Curves and
    Surfaces*, Pablo Márquez-Neila, Luis Baumela and Luis Álvarez. In IEEE
    Transactions on Pattern Analysis and Machine Intelligence (PAMI),
    2014, DOI 10.1109/TPAMI.2013.106

[2]: *Morphological Snakes*. Luis Álvarez, Luis Baumela, Pablo Márquez-Neila.
   In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2010 (CVPR10).

[3]: *Cell Shape Dynamics: From Waves to Migration*. Driscoll MK, McCann C, Kopace R, Homan T, Fourkas JT, et al.
   PLOS Computational Biology 8(3): e1002392, 2012
