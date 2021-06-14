#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <librealsense2/rsutil.h>
#include <vector>
#include <algorithm>    // std::max and min

namespace py = pybind11;

py::list to_bbox2D (const py::list& bbox3D, const py::tuple& rgb_dims, const py::dict& camints)
{
  if (bbox3D.size() != 6)
    throw py::value_error("bbox3D must be list of 6 elements");
  if (rgb_dims.size() != 2)
    throw py::value_error("Dimensions of the RGG image must be a tuple of length 2");
  float rgb_height = py::cast<float>(rgb_dims[0]);
  float rgb_width = py::cast<float>(rgb_dims[1]);

  rs2_intrinsics cam_intrinsics;
  cam_intrinsics.height = py::cast<float>(camints["height"]);
  cam_intrinsics.width = py::cast<float>(camints["width"]);
  cam_intrinsics.fx = py::cast<float>(camints["fx"]);
  cam_intrinsics.fy = py::cast<float>(camints["fy"]);
  cam_intrinsics.ppx = py::cast<float>(camints["ppx"]);
  cam_intrinsics.ppy = py::cast<float>(camints["ppy"]);
  cam_intrinsics.model = static_cast<rs2_distortion> (py::cast<int>(camints["model"]));
  py::list coeffslist = camints["coeffs"];
  cam_intrinsics.coeffs[0] = py::cast<float>(coeffslist[0]);
  cam_intrinsics.coeffs[1] = py::cast<float>(coeffslist[1]);
  cam_intrinsics.coeffs[2] = py::cast<float>(coeffslist[2]);
  cam_intrinsics.coeffs[3] = py::cast<float>(coeffslist[3]);
  cam_intrinsics.coeffs[4] = py::cast<float>(coeffslist[4]);
  
  float Apixel[2];
  float Apoint[3] = { 
    py::cast<float>(bbox3D[0]), 
    py::cast<float>(bbox3D[1]), 
    py::cast<float>(bbox3D[2])
  };
  rs2_project_point_to_pixel(Apixel, &cam_intrinsics, Apoint);
  
  float Bpixel[2];
  float Bpoint[3] = { 
    py::cast<float>(bbox3D[0]) + py::cast<float>(bbox3D[3]), 
    py::cast<float>(bbox3D[1]) + py::cast<float>(bbox3D[4]), 
    py::cast<float>(bbox3D[2]) + py::cast<float>(bbox3D[5])
  };
  rs2_project_point_to_pixel(Bpixel, &cam_intrinsics, Bpoint);
  
  // points to relative coordinates
  // Apixel[0] = Apixel[0] / cam_intrinsics.width;
  // Apixel[1] = Apixel[1] / cam_intrinsics.height;
  // Bpixel[0] = Bpixel[0] / cam_intrinsics.width;
  // Bpixel[1] = Bpixel[1] / cam_intrinsics.height;

  // points to pixels in rgb image
  // Apixel[0] = std::min( std::max(Apixel[0]*rgb_width + 0.5, 0.0), rgb_width - 1.0 );
  // Apixel[1] = std::min( std::max(Apixel[1]*rgb_height + 0.5, 0.0), rgb_height - 1.0 );
  // Bpixel[0] = std::min( std::max(Bpixel[0]*rgb_width + 0.5, 0.0), rgb_width - 1.0 );
  // Bpixel[1] = std::min( std::max(Bpixel[1]*rgb_height + 0.5, 0.0), rgb_height - 1.0 );
  // Apixel[0] = Apixel[0]*rgb_width;
  // Apixel[1] = Apixel[1]*rgb_height;
  // Bpixel[0] = Bpixel[0]*rgb_width;
  // Bpixel[1] = Bpixel[1]*rgb_height;
  
  // bbox2D = {x, y, width, height}
  std::vector<int> bbox2D = { 
    (int)Apixel[0], 
    (int)Apixel[1], 
    (int)Bpixel[0] - (int)Apixel[0], 
    (int)Bpixel[1] - (int)Apixel[1]
  };

  return py::cast(bbox2D);
}


PYBIND11_MODULE(bbox3D_to_bbox2D, m) {
  m.doc() = "pybind11 plugin"; // optional module docstring

  m.def("to_bbox2D", &to_bbox2D, "3D bounding box in the point cloud");
}
