#include <iostream>
#include <vector>

#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/region_growing_rgb.h>

void 
parseCommandLine (int argc, char** argv, std::string& cloud_fn, std::vector<int>& models, bool& downsample);

int
main (int argc, char** argv)
{
  /*********************
  * Parse command line *
  **********************/
  std::string cloud_filename;
  std::vector<int> models;
  bool downsample = true;
	parseCommandLine (argc, argv, cloud_filename, models, downsample);


  /***************** 
  * Load the cloud *
  ******************/
  pcl::PointCloud <pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud <pcl::PointXYZRGB>);
  if ( pcl::io::loadPCDFile <pcl::PointXYZRGB> (cloud_filename, *cloud) == -1 )
  {
    PCL_ERROR("Cloud reading failed.");
    return (-1);
  }


  /************************************
  * Downsample the cloud if requested *
  *************************************/
  if (downsample)
  {
    std::cout << "PointCloud before filtering has: " << cloud->points.size ()  << " points." << std::endl;
    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setInputCloud (cloud);
    vg.setLeafSize (0.002f, 0.002f, 0.002f);
    vg.filter (*cloud);
    std::cout << "PointCloud after filtering has: " << cloud->points.size ()  << " points." << std::endl;
  } else 
  {
    // Remove NaNs that could be present from not downsampling the cloud
    std::vector<int> _;
    pcl::removeNaNFromPointCloud (*cloud, *cloud, _);
  }


  /******************************
  * Region Growing segmentation *
  *******************************/
  pcl::RegionGrowingRGB<pcl::PointXYZRGB> reg;
  pcl::search::Search <pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
  reg.setInputCloud (cloud);
  reg.setSearchMethod (tree);
  reg.setDistanceThreshold (5);
  reg.setPointColorThreshold (5);
  reg.setRegionColorThreshold (5);
  reg.setMinClusterSize (1000);
  reg.setMaxClusterSize (5000);
  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);
  PCL_INFO("Found %d clusters.\n", clusters.size());


  /****************
  * Visualization *
  *****************/
  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
  pcl::visualization::PCLVisualizer viewer ("Cluster viewer");
  // Create two vertically separated viewports
  int v1 (0);
  int v2 (1);
  viewer.createViewPort (0.0, 0.0, 0.5, 1.0, v1);
  viewer.createViewPort (0.5, 0.0, 1.0, 1.0, v2);
  viewer.addText ("Original cloud", 10, 15, 16, 1, 1, 1, "info_1", v1);
  viewer.addText ("RGB region growing", 10, 15, 16, 1, 1, 1, "info_2", v2);
  viewer.addPointCloud<pcl::PointXYZRGB> (cloud, "cloud", v1);
  viewer.addPointCloud<pcl::PointXYZRGB> (colored_cloud, "colored_cloud", v2);
  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
  }

  return (0);
}

void 
parseCommandLine (int argc, char** argv, std::string& cloud_fn, std::vector<int>& models, bool& downsample)
{
  if (pcl::console::find_argument (argc, argv, "-h") >= 0)
  {
    PCL_INFO("No help for this program.\n");
    exit(EXIT_SUCCESS);
  }
  std::vector<int> fn_argspos = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (fn_argspos.size() > 1)
    PCL_ERROR("This program only works on a single cloud. Taking the first one, ignoring the rest.");
  cloud_fn = argv[fn_argspos[0]];
  // process switch if there needs to be downsampling
  if (pcl::console::find_switch(argc, argv, "--downsample"))
    downsample = true;
}