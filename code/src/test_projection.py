if __name__ == "__main__":

  import os
  import json
  import numpy as np
  from skimage import io
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
  import cv2
  from bbox3D_to_bbox2D import to_bbox2D


  pcd_dir = "../pcd/pointclouds_from_420726_09052021"
  pcd_filename = "0"
  labels_path = os.path.join(pcd_dir, pcd_filename) + ".json"
  with open(labels_path, "r") as f:
    pc_label = json.load(f)

  camintrinsics_fname = "camera_intrinsics.json"
  camintrinsics_path = os.path.join(pcd_dir, camintrinsics_fname)
  with open(camintrinsics_path, "r") as f:
    intrinsics = json.load(f)

  img_path = os.path.join(pcd_dir, pcd_filename + ".png")
  img = io.imread(img_path)
  # iterate for all unknown clusters
  bbox2D_list = []
  for cluster in pc_label["annotation"]:
    # point to relative coordinates
    x, y, width, height = to_bbox2D(cluster["bbox"], img.shape[:2], intrinsics)
    bbox2D_list.append([x, y, width, height])

  # put bboxes on the image they belong
  fig, ax = plt.subplots()
  ax.imshow(img)
  for bbox in bbox2D_list:
    rect = patches.Rectangle(
      (bbox[0], bbox[1]), 
      bbox[2], bbox[3], 
      linewidth=1, 
      edgecolor='r', 
      facecolor='none')
    ax.add_patch(rect)
  plt.show()