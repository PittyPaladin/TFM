// Code adapted from Geoffrey Biggs

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include "utils.h"

#include <iostream>

void showHelp()
{
  std::cout << "Help for this program: [NONE]" << std::endl;
}

void parseCommandLine (int argc, char** argv, std::vector<std::string>& cloud_filenames, std::string& vis_mode)
{
  if (pcl::console::find_argument (argc, argv, "-h") >= 0)
  {
    showHelp();
    exit(EXIT_SUCCESS);
  }
  std::vector<int> fn_argspos = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (fn_argspos.size() < 1)
  {
    PCL_ERROR("Need at least one point cloud.\n");
    exit(EXIT_SUCCESS);
  }
  for (int i = 0; i < fn_argspos.size(); ++i)
    cloud_filenames.push_back(argv[fn_argspos[i]]);

  // pick visualization mode desired
  if (pcl::console::find_switch (argc, argv, "--rgb"))
  {
    PCL_INFO("Visualizing RGB pointcloud\n");
    vis_mode = "rgb";
  }
  else if (pcl::console::find_switch (argc, argv, "--simple"))
  {
    PCL_INFO("Visualizing pointcloud\n");
    vis_mode = "simple";
  }

}

pcl::visualization::PCLVisualizer::Ptr rgbVis ()
{
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addCoordinateSystem (0.2);
  viewer->initCameraParameters ();
  return (viewer);
}

pcl::visualization::PCLVisualizer::Ptr simpleVis ()
{
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}


int main (int argc, char** argv)
{
  /*********************
  * Parse command line *
  **********************/
  std::vector<std::string> cloud_filenames;
  std::string vis_mode = "rgb";
  parseCommandLine (argc, argv, cloud_filenames, vis_mode);


  /****************
  * Visualization *
  *****************/
  pcl::visualization::PCLVisualizer::Ptr viewer;

  /* RGB mode */
  if (vis_mode == "rgb")
  {
    viewer = rgbVis();
    for (std::size_t idx = 0; idx < cloud_filenames.size(); ++idx)
    {
      // load the rgb pointcloud
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPtr (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr outliersCloudPtr (new pcl::PointCloud<pcl::PointXYZRGB>);
      if (pcl::io::loadPCDFile (cloud_filenames[idx], *cloudPtr) < 0)
      {
        PCL_ERROR("Error loading cloud.\n");
        exit(EXIT_FAILURE);
      }
      // sphere_filter_rgbcloud(1.2, cloudPtr, outliersCloudPtr);
      statistical_filtering (3, 50, cloudPtr, outliersCloudPtr);
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> rgb (outliersCloudPtr, 255, 0, 0); 
      viewer->addPointCloud<pcl::PointXYZRGB>(cloudPtr, "cloud" + std::to_string(idx));
      viewer->addPointCloud<pcl::PointXYZRGB>(outliersCloudPtr, rgb, "cloud_outlier" + std::to_string(idx));
    }
  }

  /* Simple mode */
  else if (vis_mode == "simple")
  {
    viewer = simpleVis();
    for (std::size_t idx = 0; idx < cloud_filenames.size(); ++idx)
    {
      // load the pointcloud
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr (new pcl::PointCloud<pcl::PointXYZ>);
      if (pcl::io::loadPCDFile (cloud_filenames[idx], *cloudPtr) < 0)
      {
        PCL_ERROR("Error loading cloud.\n");
        exit(EXIT_FAILURE);
      }
      viewer->addPointCloud<pcl::PointXYZ>(cloudPtr, "cloud" + std::to_string(idx));
    }
  }
  else
  {
    PCL_ERROR("No visualization mode provided!\n");
    exit (EXIT_SUCCESS);
  }

  // main loop of the viewer
  while (!viewer->wasStopped())
  {
    viewer->spinOnce ();
  }
    


}