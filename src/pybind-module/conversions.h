#include "opencv2/core/core.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

// https://www.programmersought.com/article/26934732792/
// cv::Mat -> numpy.ndarray
py::array_t<unsigned char> cv_mat_uint8_1c_to_numpy(cv::Mat& input) {

	py::array_t<unsigned char> dst = py::array_t<unsigned char>({input.rows, input.cols}, input.data);
	return dst;
}

py::array_t<unsigned char> cv_mat_uint8_3c_to_numpy(cv::Mat& input) {

	py::array_t<unsigned char> dst = py::array_t<unsigned char>({ input.rows, input.cols, 3 }, input.data);
	return dst;
}


// numpy.ndarray -> cv::Mat
cv::Mat numpy_uint8_1c_to_cv_mat(py::array_t<unsigned char>& input) {
 
    if (input.ndim() != 2)
        throw std::runtime_error("1-channel image must be 2 dims ");
 
    py::buffer_info buf = input.request();
 
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);
    
    return mat;
}

cv::Mat numpy_uint8_3c_to_cv_mat(py::array_t<unsigned char>& input) {
 
    if (input.ndim() != 3)
        throw std::runtime_error("3-channel image must be 3 dims ");
 
    py::buffer_info buf = input.request();
 
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);
 
    return mat;
}