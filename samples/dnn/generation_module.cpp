
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
}
