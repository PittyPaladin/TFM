// STL includes
#include <iostream>
#include <string>
// PCL includes
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>   // TicToc
#include <pcl/console/parse.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/ndt.h>
// Personal includes
#include "utils.h"

typedef pcl::PointXYZRGB PointT;
bool save_current_registration = false;

enum icpmethod {ICP, GICP, ICPNL};
void 
parseCommandLine (int argc, char** argv, std::vector<std::string>& cloud_filenames, icpmethod& icp_method, int& init_ite);
void
keyboardEventOccurred (const pcl::visualization::KeyboardEvent& event,
                       void* nothing)
{
  if (event.getKeySym () == "s" && event.keyDown ())
    save_current_registration = true;
}

int
main (int argc, char** argv)
{
  /*********************
  * Parse command line *
  **********************/
  std::vector<std::string> cloud_filenames; 
  icpmethod icp_method; 
  int iterations = 20;
  parseCommandLine (argc, argv, cloud_filenames, icp_method, iterations);
  std::string cloud_dir; 
  std::string file;
  split_filename (cloud_filenames[0], cloud_dir, file);


  /****************** 
  * Load the clouds *
  *******************/
  // iterate for all cloud filenames and store them in a vector
  std::vector<pcl::PointCloud<PointT>::Ptr> clouds_vector;
  pcl::ApproximateVoxelGrid<PointT> voxfil;
  for (std::size_t i = 0; i < cloud_filenames.size(); ++i)
  {
    pcl::PointCloud<PointT>::Ptr cloudPtr (new pcl::PointCloud<PointT>);
    if (pcl::io::loadPCDFile (cloud_filenames[i], *cloudPtr) < 0)
    {
      PCL_ERROR ("Error loading cloud %s.\n", argv[1]);
      exit(EXIT_FAILURE);
    }
    // Filter the points beyond a certain distance from the camera
    pcl::PointCloud<PointT>::Ptr outliersCloudPtr (new pcl::PointCloud<PointT>);
    sphere_filter_rgbcloud(1.2, cloudPtr, outliersCloudPtr);
    statistical_filtering (2.0, 50, cloudPtr, outliersCloudPtr);
    // Downsample all clouds using voxel grid filter
    voxfil.setLeafSize (0.005, 0.005, 0.005);
    voxfil.setInputCloud (cloudPtr);
    voxfil.filter (*cloudPtr);
    clouds_vector.push_back(cloudPtr);
  }


  /*************** 
  * Registration *
  ****************/
  // Defining a rotation matrix and translation vector
  Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity ();

  /* Apply registration algorithm */
  pcl::IterativeClosestPoint<PointT, PointT>::Ptr icpPtr (new pcl::IterativeClosestPoint<PointT, PointT>);
  if (icp_method == ICP){
    icpPtr = pcl::IterativeClosestPoint<PointT, PointT>::Ptr (new pcl::IterativeClosestPoint<PointT, PointT>);
  }
  else if (icp_method == GICP){
    icpPtr = pcl::GeneralizedIterativeClosestPoint<PointT, PointT>::Ptr (new pcl::GeneralizedIterativeClosestPoint<PointT, PointT>);
  }
  else if (icp_method == ICPNL){
    icpPtr = pcl::IterativeClosestPointNonLinear<PointT, PointT>::Ptr (new pcl::IterativeClosestPointNonLinear<PointT, PointT>);
  }
  else {
    PCL_ERROR("Method requested doesn't exist, defaulting to ICP.");
  }

  pcl::PointCloud<PointT>::Ptr sourceCloudPtr (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr targetCloudPtr (new pcl::PointCloud<PointT>);
  
  // vector to store registration transformations
  std::vector<Eigen::Matrix4f> icp_transformations_vector;
  icp_transformations_vector.push_back(Eigen::Matrix4f::Identity());
  *targetCloudPtr = *clouds_vector[0];
  icpPtr->setMaximumIterations (1000000);
  icpPtr->setEuclideanFitnessEpsilon (1e-6); 
  for (std::size_t idx = 1; idx < clouds_vector.size(); ++idx)
  {
    *sourceCloudPtr = *clouds_vector[idx];
    icpPtr->setInputSource (sourceCloudPtr);
    icpPtr->setInputTarget (targetCloudPtr);
    icpPtr->align (*sourceCloudPtr);
    if (!icpPtr->hasConverged())
    {
      PCL_WARN("ICP couldn't converge, jumping to next cloud...");
      continue;
    }
    icp_transformations_vector.push_back(icpPtr->getFinalTransformation ());
    pcl::PointCloud<PointT>::Ptr merged_cloud (new pcl::PointCloud<PointT>);
    *merged_cloud = *targetCloudPtr + *sourceCloudPtr;
    pcl::ApproximateVoxelGrid<PointT> voxfil;
    voxfil.setLeafSize (0.002, 0.002, 0.002);
    voxfil.setInputCloud (merged_cloud);
    voxfil.filter (*targetCloudPtr);
  }

  // Stat filter to output cloud
  pcl::PointCloud<PointT>::Ptr outliersCloudPtr (new pcl::PointCloud<PointT>);
  statistical_filtering (2.0, 50, targetCloudPtr, outliersCloudPtr);


  /***************
  * Data to JSON *
  ****************/
  json reg_details;
  reg_details["pcd_components"] = json::array();
  std::string _;
  std::string filename;
  for (std::size_t i = 0; i < clouds_vector.size(); ++i)
  {
    split_filename (cloud_filenames[i], _, filename);
    std::vector<float> flattened_transmat = flatten_matrix4_to_vector (icp_transformations_vector[i]);
    json pc_detail = { 
      {"filename", cloud_filenames}, 
      {"trans2reg", flattened_transmat} 
    };
    reg_details["pcd_components"].push_back(pc_detail);
  }


  /****************
  * Visualization *
  *****************/
  pcl::visualization::PCLVisualizer viewer ("viewer");
  viewer.addCoordinateSystem (0.2);
  viewer.setBackgroundColor (0, 0, 0);
  viewer.addPointCloud<PointT>(targetCloudPtr, "registered_cloud");
  viewer.registerKeyboardCallback (&keyboardEventOccurred, (void*) NULL);
  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
    if (save_current_registration)
    {
      std::string fn_tm = timestamp2string ();
      std::string cloud_savename = cloud_dir + "/reg_" + fn_tm + ".pcd";
      std::cout << "Saving current registration of point clouds as " << cloud_savename << std::endl;
      pcl::io::savePCDFileASCII(cloud_savename, *targetCloudPtr);
      // write prettified JSON to file
      std::ofstream o(cloud_dir + "/reg_" + fn_tm + ".json");
      o << std::setw(2) << reg_details << std::endl;
    }
    save_current_registration = false;
  }
}


void 
parseCommandLine (int argc, char** argv, std::vector<std::string>& cloud_filenames, icpmethod& icp_method, int& init_ite)
{
  if (pcl::console::find_argument (argc, argv, "-h") >= 0)
  {
    std::cout << "Can't provide any help" << std::endl;
    exit(EXIT_SUCCESS);
  }
  std::vector<int> fn_argspos = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (fn_argspos.size() < 2)
  {
    PCL_ERROR("Registration needs two or more point clouds.");
    exit(EXIT_SUCCESS);
  }
  for (int i = 0; i < fn_argspos.size(); ++i)
    cloud_filenames.push_back(argv[fn_argspos[i]]);

  // take method desired by the user
  std::string method;
  if (pcl::console::parse_argument (argc, argv, "--method", method) != -1)
  {
    std::transform(method.begin(), method.end(), method.begin(), ::tolower);
    if (method == "icp")
      icp_method = ICP;
    else if (method == "gicp")
      icp_method = GICP;
    else if (method == "icpnl")
      icp_method = ICPNL;
    else {
      PCL_WARN("No valid method chosen. Defaulting to ICP. ");
      icp_method = ICP;
    }
  }
  else {
    PCL_WARN("No method chosen. Defaulting to ICP. ");
    icp_method = ICP;
  }
  // number of initial iterations
  if (pcl::console::parse_argument (argc, argv, "--ite", init_ite) == -1)
    PCL_WARN("No initial iterations given. Defaulting to %d.\n", init_ite);
}
