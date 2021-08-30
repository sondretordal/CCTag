#include <iostream>
#include <string>
#include <vector>

// CCTag headers
#include "cctag/CCTag.hpp"
#include "cctag/ICCTAG.hpp"
#include "cctag/Detection.hpp"

// Boost headers
#include "boost/ptr_container/ptr_list.hpp"

// OpenCV headers
#include "opencv2/core/core.hpp"

// Pybind headers
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

// Own headers
#include "types.h"
#include "conversions.h"

namespace py = pybind11;

void drawMarkers(const boost::ptr_list<cctag::ICCTag>& markers, cv::Mat& image)
{
    // drawing settings
    const int radius{5};
    const int fontSize{1};
    const int thickness{2};
    const int fontFace{cv::FONT_HERSHEY_SIMPLEX};

    for(const auto& marker : markers)
    {
        // center of the marker
        const cv::Point center = cv::Point(marker.x(), marker.y());
        const auto rescaledOuterEllipse = marker.rescaledOuterEllipse();

        // check the status and draw accordingly, green for valid, red otherwise
        cv::Scalar color;
        if(marker.getStatus() == cctag::status::id_reliable)
        {   
            color = cv::Scalar(255, 0, 0, 255);
        }
        else
        {
            color = cv::Scalar(0, 255, 0, 255);
        }

        // draw the center
        cv::circle(image, center, radius, color, thickness);
        // write the marker ID
        cv::putText(image, std::to_string(marker.id()), center, fontFace, fontSize, color, thickness);
        // draw external ellipse
        cv::ellipse(image,
                    center,
                    cv::Size(rescaledOuterEllipse.a(), rescaledOuterEllipse.b()),
                    rescaledOuterEllipse.angle() * 180 / boost::math::constants::pi<double>(),
                    0,
                    360,
                    color,
                    thickness);
    }
}


DetectionResult cctag_detection(py::array_t<unsigned char>& input) {
    DetectionResult result = DetectionResult();
    py::buffer_info buf;
    cv::Mat color, gray;

    // Read input image
    bool inputOk = false;
    int ndim = input.ndim();
    switch (ndim) {
        case 2:
            // Grayscale image
            gray = numpy_uint8_1c_to_cv_mat(input);
            cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
            inputOk = true;
            break;

        case 3:
            // Color image
            color = numpy_uint8_3c_to_cv_mat(input);
            cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
            inputOk = true;
            break;
            
        default:
            inputOk = false;
            throw std::runtime_error("Image must be 2 or 3 in dims");
            break;
    }

    // CCtag detection settings
    const std::size_t nCrowns{3};
    cctag::Parameters params(nCrowns);
    params.setUseCuda(false);
    const int pipeId{0};
    const int frameId{0};

    // CCTag detection result
    boost::ptr_list<cctag::ICCTag> markers{};

    // Apply CCTag detection
    cctagDetection(markers, pipeId, frameId, gray, params);

    // Draw detection on color image
    drawMarkers(markers, color);

    // Convert CCTag detection list to Pybind result
    MarkerData temp = MarkerData();
    result.image = cv_mat_uint8_3c_to_numpy(color);
    for(const auto& marker : markers)
    {   
        // Fill marker data
        temp.status = marker.getStatus();
        temp.id = marker.id();
        temp.x = marker.x();
        temp.y = marker.y();
        temp.a = marker.rescaledOuterEllipse().a();
        temp.b = marker.rescaledOuterEllipse().b();
        temp.angle = marker.rescaledOuterEllipse().angle();

        // Append result
        result.markers.push_back(temp);
    }
    
    // // Show image with detection
    // cv::imshow("image", color);
    // cv::waitKey(5000);

    return result;
}

void show(std::string mTitle, cv::Mat image) {
    // Show image until window is closed
    cv::imshow(mTitle.c_str(), image);
    HWND hwnd = (HWND)cvGetWindowHandle(mTitle.c_str());
    while (IsWindowVisible(hwnd)) {
        cv::imshow(mTitle.c_str(), image);
        int key = cv::waitKey(100);

    }
}

void test()
{   
    // Read image
    cv::Mat src = cv::imread("test.png");

    // Scale down image
    cv::resize(src, src, cv::Size(), 0.5, 0.5);

    // Conevrt to gray
    cv::Mat graySrc;
    cv::cvtColor(src, graySrc, CV_BGR2GRAY);

    // set up the parameters
    const std::size_t nCrowns{3};
    cctag::Parameters params(nCrowns);

    // if you want to use GPU
    params.setUseCuda(false);
    
    // choose a cuda pipe
    const int pipeId{0};
    
    // an arbitrary id for the frame
    const int frameId{0};

    // process the image
    boost::ptr_list<cctag::ICCTag> markers{};
    cctagDetection(markers, pipeId, frameId, graySrc, params);

    // Draw markers
    drawMarkers(markers, src);

    
}

std::vector<MarkerData> testType() {
    std::vector<MarkerData> markers = std::vector<MarkerData>();
    
    for (int i = 0; i < 10; i++) {
        MarkerData marker = MarkerData();

        marker.id = i;

        markers.push_back(marker);
    }
    

    return markers;

}





PYBIND11_MODULE(PyCCTag, m) {
    m.doc() = "Python binding for CCTag detection";
    
    // Function bindings
    m.def("cctag_detection", &cctag_detection, "Detect CCTags in given image");

    // Type bindings
    py::class_<MarkerData>(m, "MarkerData")
        .def(py::init<>())
        .def_readwrite("status", &MarkerData::status)
        .def_readwrite("id", &MarkerData::id)
        .def_readwrite("x", &MarkerData::x)
        .def_readwrite("y", &MarkerData::y)
        .def_readwrite("a", &MarkerData::a)
        .def_readwrite("b", &MarkerData::b)
        .def_readwrite("angle", &MarkerData::angle);

    py::class_<DetectionResult>(m, "DetectionResult")
        .def(py::init<>())
        .def_readwrite("image", &DetectionResult::image)
        .def_readwrite("markers", &DetectionResult::markers);
}
