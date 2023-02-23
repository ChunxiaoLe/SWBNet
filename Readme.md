# SWBNet: A Stable White Balance Network for sRGB Image (AAAI2023)
Authors: Chunxiao Li, Xuejing Kang, Zhifeng Zhang, Anlong Ming*

This paper is proposed to achieve stable white balance for the sRGB images with different color temperatures by learning the color temperature-insensitive features.

# Results presentation
<p align="center">
  <img src="https://github.com/ChunxiaoLe/SWBNet/blob/master/example_images/figure1.png" alt="WB stability visualization" width="90%">
</p>
Comparing with the state-of-the-art methods, our SWBNet has a stable and superior performance for the sRGB images with different color temperatures. 


# Framework
<p align="center">
  <img src="https://github.com/ChunxiaoLe/SWBNet/blob/master/example_images/figure4.png" alt="The framework of our SWBNet" width="90%">
</p>
A. The CTIF extractor and CT-contrastive loss work together to learn the color temperature-insensitive features for achieving stable WB performance. B. The CTS-oriented transformer corrects multiple color temperature shifts differently to improve WB accuracy, especially for the multi-illumination sRGB images.

-------------------------------------------
"demo.sh": 
	You can run this code to get the white-balanced image of input.
        Changing '--input' to change different input images.
        The result are save in the folder 'result_images'.
        To use this code, do: 'sh demo.sh'
-------------------------------------------
"imgs":
        Some images are in the folder "example_images".
        These images are from Rendered Cube dataset.
        You can use this as the input image of demo.
--------------------------------------------



