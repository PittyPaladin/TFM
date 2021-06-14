//System includes
#include <iostream>
//PCL includes
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/recognition/linemod/line_rgbd.h>
#include <pcl/visualization/cloud_viewer.h>
// Personal includes
#include "utils.h"

void showHelp (char* filename);
void parseCommandLine (int argc, char** argv, std::string& cloud_fn, bool& label_reg);
void label_points_inside_bbox (
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloudPtr,
  pcl::BoundingBoxXYZ* bbox, 
  int label);

int
main (int argc, char** argv)
{
  std::string cloud_filename;
  bool label_reg = false; // label a registered cloud
  parseCommandLine (argc, argv, cloud_filename, label_reg);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr scenePtr (new pcl::PointCloud<pcl::PointXYZRGB>);
  if (pcl::io::loadPCDFile (cloud_filename, *scenePtr) < 0)
  {
    PCL_ERROR ("Error loading cloud %s.\n", cloud_filename);
    exit(EXIT_FAILURE);
  }
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr labeled_cloudPtr (new pcl::PointCloud<pcl::PointXYZRGBL> ());
  pcl::copyPointCloud(*scenePtr, *labeled_cloudPtr);

  json pc_labels;
  std::vector< std::pair<pcl::BoundingBoxXYZ, int> > bbox_vector;

  if (!label_reg)
  {
    // read JSON with labels
    std::ifstream ifs(remove_extension_from_path (cloud_filename) + ".json");
    pc_labels = json::parse(ifs);
    for (std::size_t i = 0; i < pc_labels["annotation"].size(); ++i)
    {
      pcl::BoundingBoxXYZ bbox;
      bbox.x = pc_labels["annotation"][i]["bbox"][0];
      bbox.y = pc_labels["annotation"][i]["bbox"][1];
      bbox.z = pc_labels["annotation"][i]["bbox"][2];
      bbox.width = pc_labels["annotation"][i]["bbox"][3];
      bbox.height = pc_labels["annotation"][i]["bbox"][4];
      bbox.depth = pc_labels["annotation"][i]["bbox"][5];
      bbox_vector.push_back(std::make_pair(bbox, pc_labels["annotation"][i]["category_id"]));
      label_points_inside_bbox(labeled_cloudPtr, &bbox, pc_labels["annotation"][i]["category_id"]);
    }
    
  }
  else
  {
    std::cout << "Under construction" << std::endl;
  }

  // Visualization with labels and bounding boxes
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZRGBL>(labeled_cloudPtr);
  viewer->addCoordinateSystem(0.3);
  
  // place a wireframe cube on every bounding box
  for (std::size_t i = 0; i < bbox_vector.size(); ++i)
  {
    viewer->addCube(
      bbox_vector[i].first.x, bbox_vector[i].first.x + bbox_vector[i].first.width, 
      bbox_vector[i].first.y, bbox_vector[i].first.y + bbox_vector[i].first.height, 
      bbox_vector[i].first.z, bbox_vector[i].first.z + bbox_vector[i].first.depth,
      1.0, 0.0, 0.0,
      "bbox" + std::to_string(i)
    );
    viewer->setShapeRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_REPRESENTATION, 
      pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, 
      "bbox" + std::to_string(i)
    );
  }

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce ();
  }
	

}


void 
parseCommandLine (int argc, char** argv, std::string& cloud_fn, bool& label_reg)
{
  if (pcl::console::find_argument (argc, argv, "-h") >= 0)
  {
    showHelp(argv[0]);
    exit(EXIT_SUCCESS);
  }
  std::vector<int> fn_argspos = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (fn_argspos.size() > 1)
    PCL_ERROR("This program only works on a single cloud. Taking the first one, ignoring the rest.");
  cloud_fn = argv[fn_argspos[0]];
  if (pcl::console::find_switch(argc, argv, "--label_reg"))
    label_reg = true;
}

void 
showHelp (char* filename)
{
  std::cout << std::endl;
  std::cout << "*****************************************************" << std::endl;
  std::cout << "*                                                   *" << std::endl;
  std::cout << "*             PCD Annotator - Usage Guide           *" << std::endl;
  std::cout << "*                                                   *" << std::endl;
  std::cout << "*****************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " scene_filename.pcd [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                     Show this help." << std::endl;
  std::cout << "     --label_reg:            The cloud is a registered cloud." << std::endl;
}

void
label_points_inside_bbox (
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloudPtr,
  pcl::BoundingBoxXYZ* bbox, 
  int label)
{
  int points_w_label = 0;
  for (std::size_t i = 0; i < cloudPtr->size(); ++i)
  {
    if (
      cloudPtr->points[i].x > bbox->x && cloudPtr->points[i].x < bbox->x + bbox->width &&
      cloudPtr->points[i].y > bbox->y && cloudPtr->points[i].y < bbox->y + bbox->height &&
      cloudPtr->points[i].z > bbox->z && cloudPtr->points[i].z < bbox->z + bbox->depth
    )
    {
      // START TESTING
      cloudPtr->points[i].r = 255;
      cloudPtr->points[i].g = 0;
      cloudPtr->points[i].b = 0;
      // END TESTING

      cloudPtr->points[i].label = label;
      points_w_label += 1;
    }
  }
  PCL_INFO("Class %d has %d points inside the bounding box. \n", label, points_w_label);
  // switch(label)
  // {
  //   case UNKNOWN:
  //     std::cout << "Unknown" << std::endl;
  //     break;
  //   case NONE:
  //     std::cout << "No label, no color, 0, left as is" << std::endl;
  //     break;
  //   case PEPSI:
  //     std::cout << "PEPSI" << std::endl;
  //     break;
  //   case COCACOLA:
  //     std::cout << "COCACOLA" << std::endl;
  //     break;
  //   case MILK:
  //     std::cout << "MILK" << std::endl;
  //     break;
  //   case DETERGENT:
  //     std::cout << "DETERGENT" << std::endl;
  //     break;
  //   default:
  //     PCL_ERROR("This bounding box has an undefined label!");
  // }
}