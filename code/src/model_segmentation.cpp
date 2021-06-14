// C++ includes
#include <iostream>
// PCL library includes
#include <pcl/console/parse.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/colors.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
// Personal includes
#include "utils.h"

typedef pcl::PointXYZRGB PointT;

enum model_t {PLANE, CYLINDER, CIRCLE2D, CIRCLE3D};
void 
showHelp (char *filename, bool downsample);
void 
parseCommandLine (int argc, char** argv, std::string& cloud_fn, std::vector<int>& methods, bool& downsample);


int
main (int argc, char** argv)
{
  /*********************
  * Parse command line *
  **********************/
  std::string cloud_fn;
  std::vector<int> methods;
  bool downsample = true;
  parseCommandLine(argc, argv, cloud_fn, methods, downsample);


  /***************** 
  * Load the cloud *
  ******************/
  pcl::PointCloud<PointT>::Ptr scene (new pcl::PointCloud<PointT>);
  if (pcl::io::loadPCDFile (cloud_fn, *scene) < 0)
  {
    PCL_ERROR ("Error loading cloud %s.\n", argv[1]);
    exit(EXIT_FAILURE);
  }


  /******************* 
  * Filter the cloud *
  ********************/
  pcl::PointCloud<PointT>::Ptr outliersCloudPtr (new pcl::PointCloud<PointT>);
  sphere_filter_rgbcloud (1.2, scene, outliersCloudPtr);


  /************************************
  * Downsample the cloud if requested *
  *************************************/
  if (downsample)
  {
    std::cout << "PointCloud before filtering has: " << scene->points.size ()  << " points." << std::endl;
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud (scene);
    vg.setLeafSize (0.001f, 0.001f, 0.001f);
    vg.filter (*scene);
    std::cout << "PointCloud after filtering has: " << scene->points.size ()  << " points." << std::endl;
  } else 
  {
    // Remove NaNs that could be present from not downsampling the cloud
    std::vector<int> _;
    pcl::removeNaNFromPointCloud (*scene, *scene, _);
  }

  /***************
  * Segmentation *
  ****************/
  std::vector<pcl::PointCloud<PointT>::Ptr> cluster_vector;
  for (std::size_t i = 0; i < methods.size(); ++i)
  {
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<PointT>::Ptr cloud_f (new pcl::PointCloud<PointT>);
    pcl::ExtractIndices<PointT> extract;

    // Estimate point normals
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    ne.setSearchMethod (tree);
    ne.setInputCloud (scene);
    ne.setKSearch (50);
    ne.compute (*cloud_normals);

    
    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg; 
    seg.setOptimizeCoefficients (true); // optional
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight (0.1);
    seg.setMaxIterations (100000);
    seg.setDistanceThreshold (0.06);
    seg.setInputCloud (scene);
    seg.setInputNormals (cloud_normals);
  
    // pick the ransac method
    switch (methods[i])
    {
      case 0:
        seg.setModelType(pcl::SACMODEL_PLANE);
        break;
      case 1:
        seg.setModelType(pcl::SACMODEL_CYLINDER);
        seg.setRadiusLimits (0, 0.05);
        break;
      case 2:
        seg.setModelType(pcl::SACMODEL_CIRCLE2D);
        break;
      case 3:
        seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
        break;
      default:
        seg.setModelType(pcl::SACMODEL_PLANE);
        break;
    }
    // do the segmentation
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size() == 0)
    {
      PCL_ERROR("Could not estimate a model for this scene.\n");
    }
    // extract the inliers from the input cloud
    pcl::PointCloud<PointT>::Ptr cluster (new pcl::PointCloud<PointT> ());
    extract.setInputCloud (scene);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*cluster);
    cluster_vector.push_back(cluster);
    // remove the inliers from the cloud
    extract.setNegative (true);
    extract.filter (*cloud_f);
    // cloud without the inliers is now the original scene
    *scene = *cloud_f; 
  }

  /****************
  * Visualization *
  *****************/
  pcl::visualization::PCLVisualizer viewer ("Segmentation viewer");
  viewer.addCoordinateSystem (0.2);
  viewer.setBackgroundColor (0, 0, 0);
  // display what wasn't considered inlier from the original scene
  viewer.addPointCloud<PointT> (scene, "remaining_scene");
  pcl::GlasbeyLUT colors; 
  std::string cluster_name;
  for (std::size_t i = 0; i < cluster_vector.size(); ++i)
  {
    pcl::visualization::PointCloudColorHandlerCustom<PointT> rgb (cluster_vector[i], colors.at(i).r, colors.at(i).g, colors.at(i).b); 
    cluster_name = "cluster_" + std::to_string(i);
    viewer.addPointCloud<PointT> (cluster_vector[i], rgb, cluster_name);
  }

  // maintain visualizer alive
  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
  }

  return (0);

}



/* Function definitions */
void 
parseCommandLine (int argc, char** argv, std::string& cloud_fn, std::vector<int>& methods, bool& downsample)
{
  // find cloud filename
  if (pcl::console::find_argument (argc, argv, "-h") >= 0)
  {
    showHelp(argv[0], downsample);
    exit(EXIT_SUCCESS);
  }
  std::vector<int> fn_argspos = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (fn_argspos.size() != 1)
  {
    PCL_ERROR("Need at least a valid `.pcd` point cloud.\n");
    exit(EXIT_SUCCESS);
  }
  cloud_fn = argv[fn_argspos[0]];

  // take models desired by the user in order
  if (pcl::console::parse_x_arguments (argc, argv, "--models", methods) == -1)
  {
    PCL_WARN("No arguments for --models\n");
  }
  // process switch if there needs to be downsampling
  if (pcl::console::find_switch(argc, argv, "--downsample"))
    downsample = true;
}

void showHelp (char *filename, bool downsample)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             Model segmentation - Usage Guide                            *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " scene_filename.pcd [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                     Show this help." << std::endl;
  std::cout << "     --models:               Type of model with which to do segmentation [0:PLANE, 1:CYLINDER, 2:CIRCLE2D, 3:CIRCLE3D] " << std::endl;
  std::cout << "     --downsample:           Downsample input cloud to speed computation [0:false|1:true] (default " << downsample << ")" << std::endl;
}