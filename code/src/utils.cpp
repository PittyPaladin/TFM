// C++ includes
#include <iostream>
#include <vector>
// PCL includes
#include <pcl/console/parse.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
// JSON library
#include <nlohmann/json.hpp>
using json = nlohmann::json;


void sphere_filter_rgbcloud (float R,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloudPtr, 
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr outliersCloudPtr)
{
  /* Keep points closer than R to camera. 
  Modify input scene storing only the inliers. Add the outliers to the given point cloud. */
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr inliersCloudPtr (new pcl::PointCloud<pcl::PointXYZRGB>);
  for (std::size_t idx = 0; idx < inputCloudPtr->size(); idx++) {
    pcl::PointXYZRGB point = inputCloudPtr->points[idx];
    float dist2camera = sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
    if (dist2camera <= R)
      inliersCloudPtr->push_back(point);
    else
      outliersCloudPtr->push_back(point);
  }
  PCL_INFO("Conditional sphere filter: %lu -> %lu points\n", inputCloudPtr->size(), inliersCloudPtr->size());
  copyPointCloud(*inliersCloudPtr, *inputCloudPtr);
}

void statistical_filtering (float stddev_mult,
  unsigned int mean_k,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloudPtr,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr outliersCloudPtr)
{
  /* Since input cloud and inliers cloud may be the same, copy input cloud to 
  work with it safely */
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloudCopyPtr (new pcl::PointCloud<pcl::PointXYZRGB>);
  copyPointCloud(*inputCloudPtr, *inputCloudCopyPtr); 

  // Create the filtering object
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setInputCloud (inputCloudCopyPtr);
  sor.setMeanK (mean_k);
  sor.setStddevMulThresh (stddev_mult);
  sor.filter (*inputCloudPtr);

  printf("Statistical filter: %lu -> %lu points\n", inputCloudCopyPtr->size(), inputCloudPtr->size());

  // Save outliers in another point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr statOutliersCloudPtr (new pcl::PointCloud<pcl::PointXYZRGB>); 
  sor.setNegative (true);
  sor.filter (*statOutliersCloudPtr);
  // Merge outliers using this method and the outliers that we already had
  *outliersCloudPtr += *statOutliersCloudPtr;
}

std::string timestamp2string (void)
{
  time_t now = time(0);
  tm *ltm = localtime(&now);
  std::string secs = std::string(2 - std::to_string(ltm->tm_sec).length(), '0') + std::to_string(ltm->tm_sec);
  std::string mins = std::string(2 - std::to_string(ltm->tm_min).length(), '0') + std::to_string(ltm->tm_min);
  std::string hours = std::string(2 - std::to_string(5+ltm->tm_hour).length(), '0') + std::to_string(5+ltm->tm_hour);
  std::string days = std::string(2 - std::to_string(ltm->tm_mday).length(), '0') + std::to_string(ltm->tm_mday);
  std::string months = std::string(2 - std::to_string(1+ltm->tm_mon).length(), '0') + std::to_string(1+ltm->tm_mon);
  std::string years = std::to_string(1900+ltm->tm_year);
  std::string fn_tm = secs + mins + hours + "_" + days + months + years;
  return fn_tm;
}

void split_filename (const std::string& str, std::string& path, std::string& file)
{
  std::size_t found = str.find_last_of("/\\");
  path = str.substr(0,found);
  file = str.substr(found+1);
}

void split_filenames (const std::vector<std::string>& strs, std::vector<std::string>& paths, std::vector<std::string>& files)
{
  paths.resize(strs.size());
  files.resize(strs.size());
  for (std::size_t i = 0; i < strs.size(); ++i)
  {
    std::size_t found = strs[i].find_last_of("/\\");
    paths[i] = strs[i].substr(0,found);
    files[i] = strs[i].substr(found+1);
  }
} 

std::vector<float> flatten_matrix4_to_vector (Eigen::Matrix4f eigen_mat)
{
  /* Transform a Eigen::Matrix 4f into string with 16 values separated by space */
  std::vector<float> flattened_mat;
  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      flattened_mat.push_back(eigen_mat(i,j));
    }
  }  
  return flattened_mat;
}

void cloud_segmentation (pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene, std::vector<int> methods)
{
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cluster_vector;
  for (std::size_t i = 0; i < methods.size(); ++i)
  {
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;

    // Estimate point normals
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    ne.setSearchMethod (tree);
    ne.setInputCloud (scene);
    ne.setKSearch (50);
    ne.compute (*cloud_normals);

    
    pcl::SACSegmentationFromNormals<pcl::PointXYZRGB, pcl::Normal> seg; 
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
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster (new pcl::PointCloud<pcl::PointXYZRGB> ());
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
}

void labels_from_clusters (
  std::string& cloud_filename,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr& originalCloudPtr,
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clustersPtr_vec, 
  json& pcd_label
)
{
  pcd_label["file"] = { 
    {"filename", cloud_filename},
    {"npoints", originalCloudPtr->size()},
    {"width", originalCloudPtr->width},
    {"height", originalCloudPtr->height}
  };
  pcd_label["info"] = { 
    {"contributor", "me"},
    {"year", "2021"},
    {"description", "None provided"},
  };
  pcd_label["annotation"] = json::array();
  // iterate for all clusters
  json annot;
  for (std::size_t i = 0; i < clustersPtr_vec.size(); ++i)
  {
    pcl::PointXYZRGB min_pt;
    pcl::PointXYZRGB max_pt;
    pcl::getMinMax3D (*clustersPtr_vec[i], min_pt, max_pt);
    annot = { 
      {"npoints", clustersPtr_vec[i]->size()},
      // "bbox", {x, y, z, width, height, depth}
      {"bbox", {min_pt.x, min_pt.y, min_pt.z, max_pt.x - min_pt.x, max_pt.y - min_pt.y, max_pt.z - min_pt.z} },
      // -1:unknown | 0:nothing | 1:pepsi | 2:cocacola | 3:milk | 4:detergent
      {"category_id", -1}
    };
    pcd_label["annotation"].push_back(annot);
  }
}

void save_json_labels (json json_dict, std::string path)
{
  std::ofstream o(path);
  o << std::setw(2) << json_dict << std::endl;
}

std::string remove_extension_from_path (std::string path)
{
  std::size_t lastindex = path.find_last_of("."); 
  return path.substr(0, lastindex);
}