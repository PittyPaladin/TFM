// C++ includes
#include <iostream>
// PCL library includes
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
// Personal includes
#include "utils.h"

typedef pcl::PointXYZRGB PointT;
typedef pcl::Normal NormalT;
typedef pcl::ReferenceFrame RFT;
typedef pcl::SHOT352 DescriptorT;

std::string model_filename;
std::string scene_filename;
// algorithm params
bool show_keypoints (false);
bool show_correspondences (false);
bool use_cloud_resolution (false);
bool show_normals (false);
bool txt_extension (false);
bool pcd_extension (false);
bool use_hough (true);
float model_ss (0.01f);
float scene_ss (0.03f);
float rf_rad (0.015f);
float descr_rad (0.02f);
float cg_size (0.01f);
float cg_thresh (5.0f);
int KNNSearch (10);
float filter_R (1.2);
float filter_std (2.0);

void
parseCommandLine (int argc, char *argv[]);
double
computeCloudResolution (const pcl::PointCloud<PointT>::ConstPtr &cloud);


int
main (int argc, char** argv)
{
  /*********************
  * Parse command line *
  **********************/
	parseCommandLine (argc, argv);


  /***************** 
  * Load the cloud *
  ******************/
  pcl::PointCloud <PointT>::Ptr model (new pcl::PointCloud <PointT>);
  if (pcl::io::loadPCDFile <PointT> (model_filename, *model) == -1 )
  {
    PCL_ERROR("Reading of the model cloud failed.");
    return (-1);
  }
  pcl::PointCloud <PointT>::Ptr scene (new pcl::PointCloud <PointT>);
  if (pcl::io::loadPCDFile <PointT> (scene_filename, *scene) == -1 )
  {
    PCL_ERROR("Reading of the scene cloud failed.");
    return (-1);
  }

  // Filter the points beyond a certain distance from the camera if requested
  pcl::PointCloud<PointT>::Ptr _ (new pcl::PointCloud<PointT>);
  sphere_filter_rgbcloud (filter_R, scene, _);
  // Statistical filter to eliminate noise
  statistical_filtering (filter_std, 50, scene, _);


  pcl::PointCloud<PointT>::Ptr model_keypoints (new pcl::PointCloud<PointT> ());
  pcl::PointCloud<PointT>::Ptr scene_keypoints (new pcl::PointCloud<PointT> ());
  pcl::PointCloud<DescriptorT>::Ptr model_descriptors (new pcl::PointCloud<DescriptorT> ());
  pcl::PointCloud<DescriptorT>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorT> ());


  /************************ 
  * Resolution invariance *
  *************************/
  if (use_cloud_resolution)
  {
    float resolution = static_cast<float> (computeCloudResolution (model));
    if (resolution != 0.0f)
    {
      model_ss   *= resolution;
      scene_ss   *= resolution;
      rf_rad     *= resolution;
      descr_rad  *= resolution;
      cg_size    *= resolution;
    }

    std::cout << "Model resolution:       " << resolution << std::endl;
    std::cout << "Model sampling size:    " << model_ss << std::endl;
    std::cout << "Scene sampling size:    " << scene_ss << std::endl;
    std::cout << "LRF support radius:     " << rf_rad << std::endl;
    std::cout << "SHOT descriptor radius: " << descr_rad << std::endl;
    std::cout << "Clustering bin size:    " << cg_size << std::endl << std::endl;
  }


  /****************** 
  * Compute normals *
  *******************/
  pcl::PointCloud<NormalT>::Ptr model_normals (new pcl::PointCloud<NormalT> ());
  pcl::PointCloud<NormalT>::Ptr scene_normals (new pcl::PointCloud<NormalT> ());

  pcl::NormalEstimationOMP<PointT, NormalT> norm_est;
  norm_est.setKSearch (KNNSearch);
  norm_est.setInputCloud (model);
  norm_est.compute (*model_normals);

  norm_est.setInputCloud (scene);
  norm_est.compute (*scene_normals);

  /************* 
  * Downsample *
  **************/
  // downsample the model
  pcl::UniformSampling<PointT> uniform_sampling;
  uniform_sampling.setInputCloud (model);
  uniform_sampling.setRadiusSearch (model_ss);
  uniform_sampling.filter (*model_keypoints);
  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

  // downsample the scene
  uniform_sampling.setInputCloud (scene);
  uniform_sampling.setRadiusSearch (scene_ss);
  uniform_sampling.filter (*scene_keypoints);
  std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;


  /*************************************** 
  * Compute descriptor for each keypoint *
  ****************************************/
  pcl::SHOTEstimationOMP<PointT, NormalT, DescriptorT> descr_est;
  descr_est.setRadiusSearch (descr_rad);

  descr_est.setInputCloud (model_keypoints);
  descr_est.setInputNormals (model_normals);
  descr_est.setSearchSurface (model);
  descr_est.compute (*model_descriptors);

  descr_est.setInputCloud (scene_keypoints);
  descr_est.setInputNormals (scene_normals);
  descr_est.setSearchSurface (scene);
  descr_est.compute (*scene_descriptors);

  /*********************************** 
  * Find Model-Scene Correspondences *
  ************************************/
  pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

  pcl::KdTreeFLANN<DescriptorT> match_search;
  match_search.setInputCloud (model_descriptors);

  //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
  for (std::size_t i = 0; i < scene_descriptors->size (); ++i)
  {
    std::vector<int> neigh_indices (1);
    std::vector<float> neigh_sqr_dists (1);
    if (!std::isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
    {
      continue;
    }
    int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
    if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
    {
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      model_scene_corrs->push_back (corr);
    }
  }
  std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;


  /************* 
  * Clustering *
  **************/
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
  std::vector<pcl::Correspondences> clustered_corrs;

  //  Using Hough3D
  if (use_hough)
  {
    pcl::PointCloud<RFT>::Ptr model_rf (new pcl::PointCloud<RFT> ());
    pcl::PointCloud<RFT>::Ptr scene_rf (new pcl::PointCloud<RFT> ());

    pcl::BOARDLocalReferenceFrameEstimation<PointT, NormalT, RFT> rf_est;
    rf_est.setFindHoles (true);
    rf_est.setRadiusSearch (rf_rad);

    rf_est.setInputCloud (model_keypoints);
    rf_est.setInputNormals (model_normals);
    rf_est.setSearchSurface (model);
    rf_est.compute (*model_rf);

    rf_est.setInputCloud (scene_keypoints);
    rf_est.setInputNormals (scene_normals);
    rf_est.setSearchSurface (scene);
    rf_est.compute (*scene_rf);

    //  Clustering
    pcl::Hough3DGrouping<PointT, PointT, RFT, RFT> clusterer;
    clusterer.setHoughBinSize (cg_size);
    clusterer.setHoughThreshold (cg_thresh);
    clusterer.setUseInterpolation (true);
    clusterer.setUseDistanceWeight (false);

    clusterer.setInputCloud (model_keypoints);
    clusterer.setInputRf (model_rf);
    clusterer.setSceneCloud (scene_keypoints);
    clusterer.setSceneRf (scene_rf);
    clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //clusterer.cluster (clustered_corrs);
    clusterer.recognize (rototranslations, clustered_corrs);
  }
  else
  {
    pcl::GeometricConsistencyGrouping<PointT, PointT> gc_clusterer;
    gc_clusterer.setGCSize (cg_size);
    gc_clusterer.setGCThreshold (cg_thresh);

    gc_clusterer.setInputCloud (model_keypoints);
    gc_clusterer.setSceneCloud (scene_keypoints);
    gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //gc_clusterer.cluster (clustered_corrs);
    gc_clusterer.recognize (rototranslations, clustered_corrs);
  }

  // Results
  std::cout << "Model instances found: " << rototranslations.size () << std::endl;
  for (std::size_t i = 0; i < rototranslations.size (); ++i)
  {
    std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
    std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

    // Print the rotation matrix and translation vector
    Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
    Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

    printf ("\n");
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
    printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
    printf ("\n");
    printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
  }


  /****************
  * Visualization *
  *****************/
  pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
  viewer.addPointCloud (scene, "scene_cloud");

  pcl::PointCloud<PointT>::Ptr off_scene_model (new pcl::PointCloud<PointT> ());
  pcl::PointCloud<PointT>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointT> ());

  if (show_correspondences || show_keypoints)
  {
    //  We are translating the model so that it doesn't end in the middle of the scene representation
    pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
    pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));

    pcl::visualization::PointCloudColorHandlerCustom<PointT> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
    viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
  }

  if (show_keypoints)
  {
    pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
    viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

    pcl::visualization::PointCloudColorHandlerCustom<PointT> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
    viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
  }

  if (show_normals)
  {
    int normals_density(10); // Display one normal out of normals_density
    viewer.addPointCloudNormals<PointT, NormalT> (off_scene_model, model_normals, normals_density, 0.03, "model_normals");
    viewer.addPointCloudNormals<PointT, NormalT> (scene, scene_normals, normals_density, 0.03, "scene_normals");
  }

  for (std::size_t i = 0; i < rototranslations.size (); ++i)
  {
    pcl::PointCloud<PointT>::Ptr rotated_model (new pcl::PointCloud<PointT> ());
    pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);

    std::stringstream ss_cloud;
    ss_cloud << "instance" << i;

    pcl::visualization::PointCloudColorHandlerCustom<PointT> rotated_model_color_handler (rotated_model, 255, 0, 0);
    viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());

    if (show_correspondences)
    {
      for (std::size_t j = 0; j < clustered_corrs[i].size (); ++j)
      {
        std::stringstream ss_line;
        ss_line << "correspondence_line" << i << "_" << j;
        PointT& model_point = off_scene_model_keypoints->at (clustered_corrs[i][j].index_query);
        PointT& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);

        //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
        viewer.addLine<PointT, PointT> (model_point, scene_point, 0, 255, 0, ss_line.str ());
      }
    }
  }

  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
  }

  return (EXIT_SUCCESS);

}


void
showHelp (char* filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             Correspondence Grouping Tutorial - Usage Guide              *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " model_filename.pcd scene_filename.pcd [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                     Show this help." << std::endl;
  std::cout << "     -k:                     Show used keypoints." << std::endl;
  std::cout << "     -c:                     Show used correspondences." << std::endl;
  std::cout << "     -r:                     Compute the model cloud resolution and multiply" << std::endl;
  std::cout << "                             each radius given by that value." << std::endl;
  std::cout << "     --filter_R val:         Filter points further from a certain distance from the camera (default " << filter_R << ")" << std::endl;
  std::cout << "     --filter_std val:       Statistical Filter (default " << filter_std << ")" << std::endl;
  std::cout << "     --algorithm (Hough|GC): Clustering algorithm used (default Hough)." << std::endl;
  std::cout << "     --model_ss val:         Model uniform sampling radius (default 0.01)" << std::endl;
  std::cout << "     --scene_ss val:         Scene uniform sampling radius (default 0.03)" << std::endl;
  std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;
  std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;
  std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;
  std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;
}

void
parseCommandLine (int argc, char *argv[])
{
  //Show help
  if (pcl::console::find_switch (argc, argv, "-h"))
  {
    showHelp (argv[0]);
    exit (0);
  }

  //Model & scene filenames
  std::vector<int> txt_filenames;
  std::vector<int> pcd_filenames;
  std::vector<int> filenames;
  txt_filenames = pcl::console::parse_file_extension_argument (argc, argv, ".txt");
  pcd_filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (txt_filenames.size () != 2 && pcd_filenames.size () != 2)
  {
    std::cout << "Filenames missing.\n";
    showHelp (argv[0]);
    exit (-1);
  } else if (txt_filenames.size () == 2)
  {
    filenames = txt_filenames;
    txt_extension = true;
  } else
  {
    filenames = pcd_filenames;
    pcd_extension = true;
  }
  

  model_filename = argv[filenames[0]];
  scene_filename = argv[filenames[1]];

  //Program behavior
  if (pcl::console::find_switch (argc, argv, "-k"))
  {
    show_keypoints = true;
  }
  if (pcl::console::find_switch (argc, argv, "-c"))
  {
    show_correspondences = true;
  }
  if (pcl::console::find_switch (argc, argv, "-r"))
  {
    use_cloud_resolution = true;
  }
  if (pcl::console::find_switch (argc, argv, "-n"))
  {
    show_normals = true;
  }

  std::string used_algorithm;
  if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1)
  {
    if (used_algorithm.compare ("Hough") == 0)
    {
      use_hough = true;
    }else if (used_algorithm.compare ("GC") == 0)
    {
      use_hough = false;
    }
    else
    {
      std::cout << "Wrong algorithm name.\n";
      showHelp (argv[0]);
      exit (-1);
    }
  }

  // General parameters
  pcl::console::parse_argument (argc, argv, "--filter_R", filter_R);
  pcl::console::parse_argument (argc, argv, "--filter_std", filter_std);
  pcl::console::parse_argument (argc, argv, "--model_ss", model_ss);
  pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss);
  pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad);
  pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad);
  pcl::console::parse_argument (argc, argv, "--cg_size", cg_size);
  pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh);
  pcl::console::parse_argument (argc, argv, "--KNNSearch", KNNSearch);
}

double
computeCloudResolution (const pcl::PointCloud<PointT>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<PointT> tree;
  tree.setInputCloud (cloud);

  for (std::size_t i = 0; i < cloud->size (); ++i)
  {
    if (!std::isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += std::sqrt(sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}

