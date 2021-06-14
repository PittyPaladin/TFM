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

bool next_iteration = false;
bool save_current_registration = false;
bool merge_clouds = true;

enum icpmethod {ICP, GICP, ICPNL};
void split_filename (const std::string& str, std::string& path, std::string& file);

void
print4x4Matrix (const Eigen::Matrix4d & matrix)
{
  printf ("Rotation matrix :\n");
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
  printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
  printf ("Translation vector :\n");
  printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
}
void
keyboardEventOccurred (const pcl::visualization::KeyboardEvent& event,
                       void* nothing)
{
  if (event.getKeySym () == "space" && event.keyDown ())
    next_iteration = true;
  if (event.getKeySym () == "s" && event.keyDown ())
    save_current_registration = true;
  if (event.getKeySym () == "c" && event.keyDown ())
    merge_clouds = true;
}

void 
parseCommandLine (int argc, char** argv, std::vector<std::string>& cloud_filenames, icpmethod& icp_method, int& init_ite);

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
  else{
    PCL_ERROR("Method requested doesn't exist");
  }

  icpPtr->setMaximumIterations (iterations);
  icpPtr->setInputSource (clouds_vector[1]);
  icpPtr->setInputTarget (clouds_vector[0]);
  icpPtr->setEuclideanFitnessEpsilon (1e-6); 
  // icpPtr->setMaxCorrespondenceDistance (0.05);    
  icpPtr->align (*clouds_vector[1]);
  icpPtr->setMaximumIterations (1);  // We set this variable to 1 for the next time we will call .align () function

  if (icpPtr->hasConverged ())
  {
    std::cout << "\nICP has converged, score is " << icpPtr->getFitnessScore () << std::endl;
    std::cout << "\nICP transformation " << iterations << " : cloud_icp -> cloud_in" << std::endl;
    transformation_matrix = icpPtr->getFinalTransformation ().cast<double>();
    print4x4Matrix (transformation_matrix);
  }
  else
  {
    PCL_ERROR ("\nICP has not converged.\n");
    return (-1);
  }

  /****************
  * Visualization *
  *****************/
  pcl::visualization::PCLVisualizer viewer ("ICP demo");
  // Create two vertically separated viewports
  int v1 (0);
  int v2 (1);
  viewer.createViewPort (0.0, 0.0, 0.5, 1.0, v1);
  viewer.createViewPort (0.5, 0.0, 1.0, 1.0, v2);

  // The color we will be using
  float bckgr_gray_level = 0.0;  // Black
  float txt_gray_lvl = 1.0 - bckgr_gray_level;

  // Original point cloud is white
  pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_in_color_h (clouds_vector[0], (int) 255 * txt_gray_lvl, (int) 255 * txt_gray_lvl,
                                                                             (int) 255 * txt_gray_lvl);
  viewer.addPointCloud (clouds_vector[0], cloud_in_color_h, "cloud_in_v1", v1);
  viewer.addPointCloud (clouds_vector[0], cloud_in_color_h, "cloud_in_v2", v2);

  // Transformed point cloud is green
  pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_tr_color_h (clouds_vector[0], 20, 180, 20);
  viewer.addPointCloud (clouds_vector[0], cloud_tr_color_h, "cloud_tr_v1", v1);

  // ICP aligned point cloud is red
  pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_icp_color_h (clouds_vector[1], 180, 20, 20);
  viewer.addPointCloud (clouds_vector[1], cloud_icp_color_h, "cloud_icp_v2", v2);

  // Adding text descriptions in each viewport
  viewer.addText ("White: Original point cloud\nGreen: Matrix transformed point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_1", v1);
  viewer.addText ("White: Original point cloud\nRed: ICP aligned point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_2", v2);

  std::stringstream ss;
  ss << iterations;
  std::string iterations_cnt = "ICP iterations = " + ss.str ();
  viewer.addText (iterations_cnt, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "iterations_cnt", v2);

  // Set background color
  viewer.setBackgroundColor (bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
  viewer.setBackgroundColor (bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);

  // Set camera position and orientation
  viewer.setCameraPosition (-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
  viewer.setSize (1280, 1024);  // Visualiser window size

  // Register keyboard callback :
  viewer.registerKeyboardCallback (&keyboardEventOccurred, (void*) NULL);

  int idx = 1;
  // Display the visualiser
  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();

    // The user pressed "space" :
    if (next_iteration)
    {
      // The Iterative Closest Point algorithm
      icpPtr->align (*clouds_vector[idx]);

      if (icpPtr->hasConverged ())
      {
        printf ("\033[11A");  // Go up 11 lines in terminal output.
        printf ("\nICP has converged, score is %+.0e\n", icpPtr->getFitnessScore ());
        std::cout << "\nICP transformation " << ++iterations << " : cloud_icp -> cloud_in" << std::endl;
        transformation_matrix *= icpPtr->getFinalTransformation ().cast<double>();  // WARNING /!\ This is not accurate! For "educational" purpose only!
        print4x4Matrix (transformation_matrix);  // Print the transformation between original pose and current pose

        ss.str ("");
        ss << iterations;
        std::string iterations_cnt = "ICP iterations = " + ss.str ();
        viewer.updateText (iterations_cnt, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "iterations_cnt");
        viewer.updatePointCloud (clouds_vector[1], cloud_icp_color_h, "cloud_icp_v2");
      }
      else
      {
        PCL_ERROR ("\nICP has not converged.\n");
        return (-1);
      }
    }
    if (save_current_registration)
    {
      std::string fn_tm = timestamp2string ();
      std::string cloud_savename = cloud_dir + "/reg_" + fn_tm + ".pcd";
      std::cout << "Saving current registration of point clouds as " << cloud_savename << std::endl;;
      pcl::PointCloud<PointT>::Ptr joint_cloudsPtr (new pcl::PointCloud<PointT>);
      *joint_cloudsPtr = *clouds_vector[idx] + *clouds_vector[idx + 1];
      pcl::io::savePCDFileASCII(cloud_savename, *joint_cloudsPtr);
    }
    if (merge_clouds)
    {
      std::cout << clouds_vector.size() << std::endl;
      if (idx == clouds_vector.size() - 1)
      {
        PCL_INFO("No more point clouds to registrate.\n");
      }
      else
      {
        idx += 1;
        pcl::PointCloud<PointT>::Ptr joint_cloudsPtr (new pcl::PointCloud<PointT>);
        *joint_cloudsPtr = *clouds_vector[idx] + *clouds_vector[idx + 1];
        icpPtr->setMaximumIterations (iterations);
        icpPtr->setInputSource (clouds_vector[idx + 1]);
        icpPtr->setInputTarget (joint_cloudsPtr);
        icpPtr->align (*clouds_vector[idx + 1]);
        icpPtr->setMaximumIterations (1);  // We set this variable to 1 for the next time we will call .align () function
      }
    }
    save_current_registration = false;
    next_iteration = false;
    merge_clouds = false;
  }
  return (0);
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
    PCL_WARN("No method chosen. Defaulting to ICP.");
    icp_method = ICP;
  }
  // number of initial iterations
  if (pcl::console::parse_argument (argc, argv, "--ite", init_ite) == -1)
    PCL_WARN("No initial iterations given. Defaulting to %d.\n", init_ite);
}
