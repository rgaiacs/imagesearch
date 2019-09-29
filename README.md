# ImageSearch
Search for scientific images using deep learning for computer vision
- image recommendation
- evaluation of feature extraction algorithms
- user-friendly interface

## Basic steps to run image searches:

- Install dependencies:
>> conda-env export -n imagesearch > environment.yml

- Create experiment folders:
  - *database/*: contains 1 folder per image class
  - *output/*: contains outputs of models and search results images
  - *query/*: contains query-images = 1 folder with all or 1 folder per image class
  
- Run main file to open user interface:
>> python src/pycbir.py  

## More about the project:
### Developers
- Flavio Araujo
- Romuere Silva
- Daniela Ushizima

**Reference us please!** 
This way we can continue doing social good for free!

Araujo, Silva, Ushizima, Parkinson, Hexemer, Carneiro, Medeiros, **"Reverse Image Search for Scientific Data within and beyond the Visible Spectrum", Expert Systems and Applications** 2018 [[bib]](https://dblp.uni-trier.de/pers/hb/u/Ushizima:Daniela)

This project aims to deploy a compact, yet useful package for performing content based image retrieval in python. Details about applications to images across domains can be found in this reference [[full paper]](https://www.researchgate.net/publication/325554753_Reverse_image_search_for_scientific_data_within_and_beyond_the_visible_spectrum/figures?lo=1)

![Interface PyCBIR](https://www.researchgate.net/profile/Daniela_Ushizima/publication/325554753/figure/fig5/AS:645121762680833@1530820422808/pyCBIR-interface-retrieval-options-left-with-feature-extraction-searching-method.ppm)

## Contributing
 
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D
 
## History
 
Version 0.4 (2019-09-28) - compatibility to TensorFlow 1.13
 

## License
 
The MIT License (MIT)

Copyright (c) 2015 Chris Kibble

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
