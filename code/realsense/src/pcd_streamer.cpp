// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "example.hpp"          // Include short list of convenience functions for rendering
#include <opencv2/core.hpp>     // Include OpenCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>            // std::min, std::max


int main(int argc, char * argv[]) try
{
    // Create a simple OpenGL window for rendering:
    window app(1280, 720, "RealSense Pointcloud");

    // Create an opencv named window to display the rgb color frame
    cv::namedWindow("RGB color frame", cv::WINDOW_NORMAL);

    // Construct an object to manage view state
    glfw_state app_state;
    // register callbacks to allow manipulation of the pointcloud
    register_glfw_callbacks(app, app_state);

    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    rs2::points points;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;

    // Set configuration 
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_RGB8, 30);
    // Start streaming under the specified configuration
    rs2::pipeline_profile pipe_profile = pipe.start(cfg);

    // 
    rs2::align align_to_depth(RS2_STREAM_DEPTH);

    // timestamp associated to current run
    std::string tmstmp = timestamp2string();
    // id associated to each image
    int img_id = 0;
    while (app) // Application still alive?
    {
        // Wait for the next set of frames from the camera
        auto frames = pipe.wait_for_frames();

        auto color_frame = frames.get_color_frame();

        // For cameras that don't have RGB sensor, we'll map the pointcloud to infrared instead of color
        if (!color_frame)
            color_frame = frames.get_infrared_frame();

        // Tell pointcloud object to map to this color frame
        pc.map_to(color_frame);

        auto depth_frame = frames.get_depth_frame();

        // Generate the pointcloud and texture mappings
        points = pc.calculate(depth_frame);

        // Upload the color frame to OpenGL
        app_state.tex.upload(color_frame);

        // Draw the pointcloud
        draw_pointcloud(app.width(), app.height(), app_state, points);

        // query frame size (width and height)
        const int w = color_frame.as<rs2::video_frame>().get_width();
        const int h = color_frame.as<rs2::video_frame>().get_height();
        cv::Mat image(cv::Size(w, h), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        // Display the rgb color frame
        cv::imshow("RGB color frame", image);
        cv::waitKey(25); // wait 25 ms
        
        // Save pointcloud as pcd?
        if (app.save_as_pcd)
        {
            // save camera intrinsics in json
            rs2_intrinsics intrinsics = pipe_profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
            save_camera_intrinsics (intrinsics, tmstmp);
            
            // save the pointcloud
            save_rs2points_to_pcd(app, points, color_frame, tmstmp, img_id);
            
            // save the RGB image
            // save_cvMat_to_png(image, tmstmp, img_id);
            frames = align_to_depth.process(frames);
            save_aligned_depth_frame_to_png(frames, tmstmp, img_id);

            img_id += 1;
        }
            
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
