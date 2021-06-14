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
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
// Personal includes
#include "utils.h"

typedef pcl::PointXYZRGB PointT;
enum model_t {PLANE, CYLINDER, CIRCLE2D, CIRCLE3D};
bool save_clusters = false;
bool save_labels = false;

void showHelp (char* filename, bool downsample, float filter_R, float filter_std, double cluster_tolerance, int max_cluster_size, int min_cluster_size);
void parseCommandLine (int argc, char** argv, std::string& cloud_fn, std::vector<int>& models, bool& downsample, float& filter_R, float& filter_std, double& cluster_tolerance, int& max_cluster_size, int& min_cluster_size);
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent& event, void* nothing)
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
  float filter_std = 2.0;
  double cluster_tolerance = 0.02;
  int max_cluster_size = 50000000;
  int min_cluster_size = 100;
  parseCommandLine (argc, argv, cloud_filename, models, 
    downsample, filter_R, filter_std, cluster_tolerance, max_cluster_size, min_cluster_size);


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
  statistical_filtering(filter_std, 50, scene, outliersCloudPtr);


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


  /*********************************
  * Euclidean Cluster segmentation *
  **********************************/
  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
  tree->setInputCloud (scene);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance (cluster_tolerance);
  ec.setMinClusterSize (min_cluster_size);
  ec.setMaxClusterSize (max_cluster_size);
  ec.setSearchMethod (tree);
  ec.setInputCloud (scene);
  ec.extract (cluster_indices);
  PCL_INFO("Found %d clusters.\n", cluster_indices.size());

  // Place each cluster in a separate point cloud
  std::vector<pcl::PointCloud<PointT>::Ptr> clusters_vector; 
  clusters_vector.resize(cluster_indices.size());
  for (std::size_t i = 0; i < cluster_indices.size (); i++)
  {
    pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);
    for (std::vector<int>::const_iterator pit = cluster_indices[i].indices.begin (); pit != cluster_indices[i].indices.end (); ++pit)
    {
      cloud_cluster->points.push_back (scene->points[*pit]);
    }
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;
    clusters_vector[i] = cloud_cluster;
  }



  /****************
  * Visualization *
  *****************/
  pcl::visualization::PCLVisualizer viewer ("Cluster viewer");
  viewer.registerKeyboardCallback (&keyboardEventOccurred, (void*) NULL);
  std::string cluster_name;
  pcl::GlasbeyLUT colors; 
  for (std::size_t i = 0; i < clusters_vector.size(); ++i)
  {
    cluster_name = "cloud" + std::to_string(i);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> rgb (clusters_vector[i], colors.at(i).r, colors.at(i).g, colors.at(i).b); 
    viewer.addPointCloud<PointT>(clusters_vector[i], rgb, cluster_name);
  }
  
  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();

    // user wants to save clusters of point clouds as separate clouds
    if (save_clusters) {
      PCL_INFO("Saving the clusters as clouds in separate files.\n");
      for (std::size_t i = 0; i < clusters_vector.size(); ++i) {
        pcl::io::savePCDFileASCII("cluster" + std::to_string(i) + ".pcd", *clusters_vector[i]);
      }
    save_clusters = false;
    }
    if (save_labels)
    {
      // save segmentations in a json labels file
      json json_labels;
      labels_from_clusters (cloud_filename, scene, clusters_vector, json_labels);
      save_json_labels (json_labels, remove_extension_from_path(cloud_filename) + ".json");
    }
    save_labels = false;
  }


}

void 
parseCommandLine (int argc, char** argv, std::string& cloud_fn, std::vector<int>& models, bool& downsample, float& filter_R, float& filter_std, double& cluster_tolerance, int& max_cluster_size, int& min_cluster_size)
{
  if (pcl::console::find_argument (argc, argv, "-h") >= 0)
  {
    showHelp(argv[0], downsample, filter_R, filter_std, cluster_tolerance, max_cluster_size, min_cluster_size);
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
  // Statistical filter
  pcl::console::parse_argument (argc, argv, "--filter_std", filter_std);
  // Take smoothness threshold
  pcl::console::parse_argument (argc, argv, "--cluster_t", cluster_tolerance);
  // Min and max cluster size
  pcl::console::parse_argument (argc, argv, "--max_size", max_cluster_size);
  pcl::console::parse_argument (argc, argv, "--min_size", min_cluster_size);
}

void 
showHelp (char* filename, bool downsample, float filter_R, float filter_std, double cluster_tolerance, int max_cluster_size, int min_cluster_size)
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
  std::cout << "     --filter_std:           Statistical Filter (default " << filter_std << ")" << std::endl;
  std::cout << "     --cluster_t:            Cluster tolerance (default " << cluster_tolerance << ")." << std::endl;
  std::cout << "     --max_size:             Maximum cluster size (default " << max_cluster_size << ")." << std::endl;
  std::cout << "     --min_size:             Minimum cluster size (default " << min_cluster_size << ")." << std::endl;
}