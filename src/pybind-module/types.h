#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

struct MarkerData {
    int status;
    int id;
    int x;
    int y;
    int a;
    int b;
    float angle;
};

struct DetectionResult
{
    pybind11::array_t<unsigned char> image;
    std::vector<MarkerData> markers = std::vector<MarkerData>();
};

