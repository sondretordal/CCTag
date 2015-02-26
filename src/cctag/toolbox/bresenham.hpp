#ifndef _CCTAG_TOOLBOX_BRESENHAM_HPP_
#define	_CCTAG_TOOLBOX_BRESENHAM_HPP_

#include <boost/gil/gil_all.hpp>
#include <cctag/imageCut.hpp>

namespace popart
{
namespace toolbox
{

void bresenham( const boost::gil::gray8_view_t & sView, const popart::Point2dN<int>& p, const popart::Point2dN<float>& dir, const std::size_t nmax, ImageCut & cut );

}	
}

#endif
