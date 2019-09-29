ImageSearch
===========

Search for scientific images using deep learning for computer vision - image recommendation - evaluation of feature extraction algorithms - user-friendly interface

Basic steps to run image searches:
----------------------------------

-	Install dependencies and activate env: >> conda env create -n imagesearch -f environment.yml >> conda activate imagesearch

-	Create experiment folders:

	-	*database/*: contains 1 folder per image class
	-	*output/*: contains outputs of models and search results images
	-	*query/*: contains query-images = 1 folder with all or 1 folder per image class

-	Run main file to open user interface: >> cd src/ >> python pycbir.py

-	Need a dataset? Check below, it will work in your laptop:

	-	[[Cells]](https://drive.google.com/open?id=13Ee5D7IT4ZU63Hext3ZTqtExlRvgmJwM)

More about the project:
-----------------------

### Developers

-	Flavio Araujo
-	Romuere Silva
-	Daniela Ushizima

**Reference us please!** This way we can continue doing social good for free!

**Araujo, Silva, Ushizima, Parkinson, Hexemer, Carneiro, Medeiros, "Reverse Image Search for Scientific Data within and beyond the Visible Spectrum", Expert Systems and Applications 2018** [[bib]](https://dblp.uni-trier.de/pers/hb/u/Ushizima:Daniela)

This project aims to deploy a compact, yet useful package for performing content based image retrieval in python. Details about applications to images across domains can be found in this reference [[full paper]](https://www.researchgate.net/publication/325554753_Reverse_image_search_for_scientific_data_within_and_beyond_the_visible_spectrum/figures?lo=1)

![Interface PyCBIR](https://www.researchgate.net/profile/Daniela_Ushizima/publication/325554753/figure/fig5/AS:645121762680833@1530820422808/pyCBIR-interface-retrieval-options-left-with-feature-extraction-searching-method.ppm)
