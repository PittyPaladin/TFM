#pragma once
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Labels category_id
#define UNKNOWN -1
#define NONE 0
#define PEPSI 1
#define COCACOLA 2
#define MILK 3
#define DETERGENT 4

void 
sphere_filter_rgbcloud (float R,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloudPtr, 
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr outliersCloudPtr);

void 
statistical_filtering (float stddev_mult,
  unsigned int mean_k,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloudPtr,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr outliersCloudPtr);

std::string 
timestamp2string (void);

void 
split_filename (const std::string& str, std::string& path, std::string& file);
void
split_filenames (const std::vector<std::string>& strs, std::vector<std::string>& paths, std::vector<std::string>& files);

std::vector<float> 
flatten_matrix4_to_vector (Eigen::Matrix4f eigen_mat);

void
cloud_segmentation (pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene, std::vector<int> methods);

void labels_from_clusters (
  std::string& cloud_filename,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr& originalCloudPtr,
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clustersPtr_vec, 
  json& pcd_label
);

void 
save_json_labels (json json_dict, std::string path);

std::string
remove_extension_from_path (std::string path);