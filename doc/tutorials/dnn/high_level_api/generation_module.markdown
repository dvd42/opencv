High Level API: Generation Model
================================

### Introduction
----------------
In this tutorial we will present how easy it is to run inference on a model using the new OpenCV High_Level_AP

Here is a simple snippet of the code in C++ with which you can run inference on a generation model. It is worth noticing that
by generation model I mean any model who's output is a generated or reconstructed image (e.g: GANs, Style Transfer models,
Image Colorization models, etc)

This particular model was trained on this image


![mosaic image](images/mosaic.jpg)


```cpp

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace dnn;

int main()
{

    // Load our sample image
    Mat image;
    image = imread(samples::findFile("style_sample.jpg"));

    GenerationModel model(samples::findFile("style.onnx")); // create our model

    // Define the transformations that we need to apply to our image
    Size size{512, 512}; // resize

    // Swap Red and Blue channels since OpenCV loads images in BGR
    // and the network was trained with RGB images
    bool swapRB = true;

    // 1/std to normalize the image
    // In this case the input doesn't need to be normalized for style transfer
    double scale = 1.0;
    Scalar mean = Scalar(); // mean to substract (e.g: 103.939, 116.779, 123.68)

    // Set the transformations we want to apply
    model.setInputParams(scale, size, mean, swapRB);

    // Network Forward pass
    std::vector<Mat> images;
    images = model.forward(image);

    //Display Image
    imshow("styled_image.png", images[0]);
    waitKey(0);

    return 0;
}
```

Here are the results:

<p float="left">
<img src="images/styled_image.png" width="256" height="256"/> <img src="images/coffe_cup.jpg" width="256" height="256"/>
</p>
And thats it! In just a few lines we have sucessfully ran our generative model
