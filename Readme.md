# SWBNet: A Stable White Balance Network for sRGB Image (AAAI2023)
Authors: Chunxiao Li, Xuejing Kang, Zhifeng Zhang, Anlong Ming*

This paper is proposed to achieve stable white balance for the sRGB images with different color temperatures by learning the color temperature-insensitive features.

# Results presentation
<p align="center">
  <img src="https://github.com/ChunxiaoLe/SWBNet/blob/master/example_images/figure1.png" alt="WB stability visualization1" width="90%">
</p>
Comparing with the state-of-the-art methods, our SWBNet has a stable and superior performance for the sRGB images with different color temperatures. 


# Network Structure
<p align="center">
  <img src="https://github.com/ChunxiaoLe/SWBNet/blob/master/example_images/figure1.png" alt="WB stability visualization1" width="90%">
</p>

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



