// C++ includes
#include <iostream>
// PCL library includes
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/common/colors.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
// Personal includes
#include "utils.h"

typedef pcl::PointXYZRGB PointT;
enum model_t {PLANE, CYLINDER, CIRCLE2D, CIRCLE3D};
bool save_clusters = false;
bool save_labels = false;

void showHelp (char* filename, bool downsample, float filter_R, float smoothness_threshold, float curvature_threshold);
void parseCommandLine (int argc, char** argv, std::string& cloud_fn, std::vector<int>& models, bool& downsample, float& filter_R, float& smoothness_threshold, float& curvature_threshold);
void
keyboardEventOccurred (const pcl::visualization::KeyboardEvent& event,
  void* nothing)
{
  if (event.getKeySym () == "s" && event.keyDown ())
    save_clusters = true;
  if (event.getKeySym () == "l" && event.keyDown ())
    save_labels = true;
}



int
main (int argc, char** argv)
{
  /*********************
  * Parse command line *
  **********************/
	std::string cloud_filename;
  std::vector<int> models;
  bool downsample = true;
  float filter_R = 1.2;
  float smoothness_threshold = 5.0;
  float curvature_threshold = 1.0;
	parseCommandLine (argc, argv, cloud_filename, models, downsample, filter_R, smoothness_threshold, curvature_threshold);


  /***************** 
  * Load the cloud *
  ******************/
	pcl::PointCloud<PointT>::Ptr scene (new pcl::PointCloud<PointT>);
  if (pcl::io::loadPCDFile (cloud_filename, *scene) < 0)
  {
    PCL_ERROR ("Error loading cloud %s.\n", cloud_filename);
    exit(EXIT_FAILURE);
  }
  // Filter the input cloud
  pcl::PointCloud<PointT>::Ptr outliersCloudPtr (new pcl::PointCloud<PointT>);
  sphere_filter_rgbcloud(filter_R, scene, outliersCloudPtr);


  /************************************
  * Downsample the cloud if requested *
  *************************************/
  if (downsample)
  {
    std::cout << "PointCloud before filtering has: " << scene->points.size ()  << " points." << std::endl;
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud (scene);
    vg.setLeafSize (0.005f, 0.005f, 0.005f);
    vg.filter (*scene); 
    std::cout << "PointCloud after filtering has: " << scene->points.size ()  << " points." << std::endl;
  } else 
  {
    // Remove NaNs that could be present from not downsampling the cloud
    std::vector<int> _;
    pcl::removeNaNFromPointCloud (*scene, *scene, _);
  }


  /*********************************************
  * Preliminar model segmentation if requested *
  **********************************************/
  cloud_segmentation(scene, models);


  /******************************
  * Region Growing segmentation *
  *******************************/
  // compute cloud normals
  pcl::search::Search<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<PointT, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (scene);
  normal_estimator.setKSearch (100);
  normal_estimator.compute (*normals);
  // perform the region growing segmentation from the cloud and its normals
  pcl::RegionGrowing<PointT, pcl::Normal> reg;
  reg.setMinClusterSize (100);
  reg.setMaxClusterSize (1000000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (30);
  reg.setInputCloud (scene);
  reg.setInputNormals (normals);
  reg.setSmoothnessThreshold (smoothness_threshold / 180.0 * M_PI);
  reg.setCurvatureThreshold (curvature_threshold);
  std::vector <pcl::PointIndices> clusters_indices;
  reg.extract (clusters_indices);
  PCL_INFO("Found %d clusters.\n", clusters_indices.size());

  // put the clusters into separate point clouds
  std::vector<pcl::PointCloud<PointT>::Ptr> clusters_vector; 
  clusters_vector.resize(clusters_indices.size());
  for (std::size_t i = 0; i < clusters_indices.size(); ++i)
  {
    pcl::PointCloud<PointT>::Ptr cluster_cloud (new pcl::PointCloud<PointT>);
    for (std::vector<int>::const_iterator pit = clusters_indices[i].indices.begin (); pit != clusters_indices[i].indices.end (); ++pit)
    {
      cluster_cloud->points.push_back (scene->points[*pit]);
    }
    cluster_cloud->width = cluster_cloud->points.size ();
    cluster_cloud->height = 1;
    cluster_cloud->is_dense = true;
    clusters_vector[i] = cluster_cloud;

  }


  /****************
  * Visualization *
  *****************/
  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
  pcl::visualization::PCLVisualizer viewer ("Cluster viewer");
  viewer.addPointCloud(colored_cloud);
  viewer.registerKeyboardCallback (&keyboardEventOccurred, (void*) NULL);
  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();

    // user wants to save clusters of point clouds as separate clouds
    if (save_clusters) {
      PCL_INFO("Saving the cluster in separate files.\n");
      for (std::size_t i = 0; i < clusters_vector.size(); ++i) {
        pcl::io::savePCDFileASCII("cluster" + std::to_string(i) + ".pcd", *clusters_vector[i]);
      }
    save_clusters = false;
    }

    if (save_labels)
    {
      // save segmentations in a json labels file
      PCL_INFO("Saving labels in JSON file.\n");
      json json_labels;
      labels_from_clusters (cloud_filename, scene, clusters_vector, json_labels);
      save_json_labels (json_labels, remove_extension_from_path(cloud_filename) + ".json");
    }
    save_labels = false;
  }


}

void 
parseCommandLine (int argc, char** argv, std::string& cloud_fn, std::vector<int>& models, bool& downsample, float& filter_R, float& smoothness_threshold, float& curvature_threshold)
{
  if (pcl::console::find_argument (argc, argv, "-h") >= 0)
  {
    showHelp(argv[0], downsample, filter_R, smoothness_threshold, curvature_threshold);
    exit(EXIT_SUCCESS);
  }
  std::vector<int> fn_argspos = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (fn_argspos.size() > 1)
    PCL_ERROR("This program only works on a single cloud. Taking the first one, ignoring the rest.");
  cloud_fn = argv[fn_argspos[0]];
  // take models desired by the user in order
  if (!pcl::console::parse_x_arguments (argc, argv, "--models", models))
  {
    PCL_WARN("No arguments for --models.\n");
  }
  // process switch if there needs to be downsampling
  if (pcl::console::find_switch(argc, argv, "--downsample"))
    downsample = true;
  // Take filter by distance from camera
  pcl::console::parse_argument (argc, argv, "--filter_R", filter_R);
  // Take smoothness threshold
  pcl::console::parse_argument (argc, argv, "--smooth_t", smoothness_threshold);
  // Take curvature threshold
  pcl::console::parse_argument (argc, argv, "--curv_t", curvature_threshold);
}

void 
showHelp (char *filename, bool downsample, float filter_R, float smoothness_threshold, float curvature_threshold)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             Region growing segmentation - Usage Guide                   *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " scene_filename.pcd [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                     Show this help." << std::endl;
  std::cout << "     --models:               Type of model with which to do segmentation [0:PLANE, 1:CYLINDER, 2:CIRCLE2D, 3:CIRCLE3D] before actually computing the region growing." << std::endl;
  std::cout << "     --downsample:           Downsample input cloud to speed computation (default " << downsample << ")." << std::endl;
  std::cout << "     --filter_R:             Filter point cloud by distance R from the camera (default " << filter_R << ")." << std::endl;
  std::cout << "     --smooth_t:             Smoothness threshold (default " << smoothness_threshold << ")." << std::endl;
}