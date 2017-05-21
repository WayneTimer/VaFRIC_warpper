#include <cstdio>
#include <string>
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <ros/package.h>
#include <geometry_msgs/PoseStamped.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <boost/thread.hpp>

#include "VaFRIC/VaFRIC.h"

using namespace std;

#define DOUBLE_EPS 1e-6
#define PYRDOWN_LEVEL 3

class CALI_PARA
{
public:
    double fx[PYRDOWN_LEVEL],fy[PYRDOWN_LEVEL];
    double cx[PYRDOWN_LEVEL],cy[PYRDOWN_LEVEL];
    int width[PYRDOWN_LEVEL],height[PYRDOWN_LEVEL];
};

class DEPTH_DATA
{
public:
    Eigen::MatrixXd depth;
    double stamp;
};

#define HZ 200
#define SAMPLE 0
#define IMG_FIRST_NO 0
#define IMG_CNT 100

const char dir_folder[] = "/home/timer/VaFRIC_datasets/200Hz";
CALI_PARA cali;
vector<DEPTH_DATA> depth_vec;
ros::Publisher pub_cur_pose, pub_image;
ros::Time time_stamp;
ros::Publisher pub_my_depth, pub_gt_depth, pub_error_map;

void vafric_init()
{
    depth_vec.clear();

    ros::NodeHandle nh("~");
    if (!nh.getParam("fx",cali.fx[0]) ||
        !nh.getParam("fy",cali.fy[0]) ||
        !nh.getParam("cx",cali.cx[0]) ||
        !nh.getParam("cy",cali.cy[0]) ||
        !nh.getParam("width",cali.width[0]) ||
        !nh.getParam("height",cali.height[0])
       )
        ROS_ERROR("Read rosparam error !");

    ROS_INFO("fx = %lf, fy = %lf",cali.fx[0], cali.fy[0]);
    ROS_INFO("cx = %lf, cy = %lf",cali.cx[0], cali.cy[0]);
    ROS_INFO("height = %d, width = %d",cali.height[0], cali.width[0]);

    // pyr_down: (fx,fy,cx,cy)
    for (int level=1;level<PYRDOWN_LEVEL;level++)
    {
        cali.fx[level] = cali.fx[0] / (1<<level);
        cali.fy[level] = cali.fy[0] / (1<<level);
        cali.cx[level] = (cali.cx[0]+0.5) / (1<<level) - 0.5;
        cali.cy[level] = (cali.cy[0]+0.5) / (1<<level) - 0.5;
        cali.width[level] = cali.width[0] >> level;
        cali.height[level] = cali.height[0] >> level;
    }
}

void vafric_input(int i, cv::Mat& img, Eigen::MatrixXd& depth,
                  geometry_msgs::Pose& pose,
                  ros::Time& stamp)
{
    int height,width;
    height = cali.height[0];
    width = cali.width[0];

    dataset::vaFRIC dataset(dir_folder,width,height,
                            cali.cx[0],cali.cy[0],
                            cali.fx[0],cali.fy[0]);


    // read img: type = CV_8UC1
    img = dataset.getPNGImage(i,SAMPLE,0);

    // prepare img depth
    depth = Eigen::MatrixXd::Zero(height,width);
    FILE *depth_file;
    char file_path[100];
    sprintf(file_path,"%s/scene_%02d_%04d.depth",dir_folder,SAMPLE,i);
    depth_file = fopen(file_path,"r");

    int u,v;
    u = 0;
    v = 0;
    for (int j=0;j<height*width;j++)
    {
        double t1,t2,z;
        if (fscanf(depth_file,"%lf",&z)!=1)
        {
            ROS_ERROR("Error: Read depth_file error !");
            exit(0);
        }

        t1 = (u-cali.cx[0])/cali.fx[0];
        t1*=t1;
        t2 = (v-cali.cy[0])/cali.fy[0];
        t2*=t2;
        depth(v,u) = z / sqrt(t1+t2+1.0);

        u++;
        if (u>=width)
        {
            u^=u;
            v++;
        }
    }
    fclose(depth_file);

    // cam_pose: p,q    (R_k^0, T_k^0)
    Eigen::MatrixXf RT;
    RT = dataset.computeTpov_cam(i,SAMPLE);
    Eigen::Matrix3f R,tmpR;
    Eigen::Vector3f T;
    tmpR = RT.topLeftCorner<3,3>();
    T = RT.col(3);

    R = tmpR;

    Eigen::Quaternionf q = Eigen::Quaternionf(R);

    Eigen::Vector3d gt_p;
    Eigen::Quaterniond gt_q;
    gt_p = T.cast<double>();
    gt_q = q.cast<double>();

ROS_INFO("gt_q: (w,x,y,z) = (%.2lf,%.2lf,%.2lf,%.2lf)",
            gt_q.w(),gt_q.x(),gt_q.y(),gt_q.z()
        );

    pose.position.x = gt_p(0);
    pose.position.y = gt_p(1);
    pose.position.z = gt_p(2);
    pose.orientation.w = gt_q.w();
    pose.orientation.x = gt_q.x();
    pose.orientation.y = gt_q.y();
    pose.orientation.z = gt_q.z();    

    stamp = stamp + ros::Duration(1.0/HZ);
}

double cal_density(Eigen::MatrixXd& depth)
{
    double ans = 0.0;
    int height = cali.height[0];
    int width = cali.width[0];

    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
            if (depth(v,u) < 700.0)
                ans = ans + 1.0;
    ans = ans / (height*width);
    return ans;
}

double cal_error(Eigen::MatrixXd& my_depth, Eigen::MatrixXd& gt_depth, Eigen::MatrixXd& error_map)
{
    double ans = 0.0;
    double cnt = 0.0;
    int height = cali.height[0];
    int width = cali.width[0];

    error_map = Eigen::MatrixXd::Zero(height,width);

    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
            if (my_depth(v,u) < 700.0)
            {
                error_map(v,u) = fabs(my_depth(v,u) - gt_depth(v,u)) / fabs(gt_depth(v,u));
                ans = ans + error_map(v,u);
                cnt = cnt + 1.0;
            }
            else
                error_map(v,u) = -1.0;
    ans = ans / cnt;
    return ans;
}

double cal_outlier_ratio(Eigen::MatrixXd& error_map)
{
    double ans = 0.0;
    double cnt = 0.0;
    int height = cali.height[0];
    int width = cali.width[0];

    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
            if (error_map(v,u) > 0.0)
            {
                cnt = cnt + 1.0;
                if (error_map(v,u) > 0.10)
                    ans = ans + 1.0;
            }
    ans = ans / cnt;
    return ans;
}

void show_depth(const char* window_name, Eigen::MatrixXd &depth, ros::Publisher &pub)
{
    int height,width;
    height = depth.rows();
    width = depth.cols();

    double max_dep = -1;
    double avg_dep = 0.0;
    double min_dep = -1;
    int avg_cnt = 0;
    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
            if (depth(v,u) < 700.0f)
            {
                avg_dep += depth(v,u);
                avg_cnt++;
                if (depth(v,u) > max_dep)
                    max_dep = depth(v,u);
                if (min_dep<0 || depth(v,u) < min_dep)
                    min_dep = depth(v,u);
            }
    ROS_WARN("%s: %d x %d, max_dep = %lf, min_dep = %lf, avg_dep = %lf",window_name, height, width, max_dep, min_dep, avg_dep/avg_cnt);
    cv::Mat show_img = cv::Mat::zeros(height,width,CV_8UC1);
    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
            if (depth(v,u) < max_dep)
            {
                double t = depth(v,u)*255.0 / max_dep;
                if (t<0) t = 0.0;
                if (t>255) t = 255.0;
                show_img.at<uchar>(v,u) = (uchar)t;
            }
    cv::Mat depth_img;
    cv::applyColorMap(show_img, depth_img, cv::COLORMAP_JET);

    {
        cv_bridge::CvImage out_msg;
        out_msg.header.stamp = time_stamp;
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        out_msg.image = depth_img.clone();
        pub.publish(out_msg.toImageMsg());
    }
    //cv::imshow(window_name, depth_img);
}

void show_error_map(const char* window_name, Eigen::MatrixXd &error_map, ros::Publisher &pub)
{
    int height,width;
    height = error_map.rows();
    width = error_map.cols();

    double max_dep = -1;
    double avg_dep = 0.0;
    double min_dep = -1;
    int avg_cnt = 0;
    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
            if (error_map(v,u) > 0.0)
            {
                avg_dep += error_map(v,u);
                avg_cnt++;
                if (error_map(v,u) > max_dep)
                    max_dep = error_map(v,u);
                if (min_dep<0 || error_map(v,u) < min_dep)
                    min_dep = error_map(v,u);
            }
    ROS_WARN("%s: %d x %d, max_error = %lf%%, min_error = %lf%%, avg_error = %lf%%",window_name, height, width, max_dep*100.0, min_dep*100.0, avg_dep*100.0/avg_cnt);

    max_dep = 0.20;

    cv::Mat show_img = cv::Mat::zeros(height,width,CV_8UC1);
    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
            if (error_map(v,u) > 0.0)
            {
                double t = error_map(v,u)*255.0 / max_dep;
                if (t<0) t = 0.0;
                if (t>255) t = 255.0;
                show_img.at<uchar>(v,u) = (uchar)t;
            }
    cv::Mat depth_img;
    cv::applyColorMap(show_img, depth_img, cv::COLORMAP_JET);

    {
        cv_bridge::CvImage out_msg;
        out_msg.header.stamp = time_stamp;
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        out_msg.image = depth_img.clone();
        pub.publish(out_msg.toImageMsg());
    }
}

void depth_callback(const sensor_msgs::ImageConstPtr msg)
{
    double t1 = msg->header.stamp.toSec();
    int i,l;
    l = depth_vec.size();

    for (i=0;i<l;i++)
    {
        double t;
        t = fabs(depth_vec[i].stamp - t1);
        if (t < DOUBLE_EPS)
            break;
    }

    cv_bridge::CvImagePtr depth_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
    Eigen::MatrixXd my_depth;
    cv::cv2eigen(depth_ptr->image,my_depth);

    double density = cal_density(my_depth);
    Eigen::MatrixXd error_map;
    double error = cal_error(my_depth, depth_vec[i].depth, error_map);
    double outlier_ratio = cal_outlier_ratio(error_map);

    ROS_WARN("density = %lf%%, error = %lf%%, outlier_ratio (10%) = %lf%%", density*100.0, error*100.0,outlier_ratio*100.0);

    show_depth("my_depth",my_depth,pub_my_depth);
    show_depth("gt_depth",depth_vec[i].depth,pub_gt_depth);
    show_error_map("error_map",error_map,pub_error_map);
    //cv::waitKey(0);
}

void main_thread()
{
    for (int i=IMG_FIRST_NO;i<IMG_CNT;i++)
    {
        DEPTH_DATA depth_data;
        cv::Mat img;
        geometry_msgs::PoseStamped cur_pose;

        vafric_input(i, img, depth_data.depth, cur_pose.pose, time_stamp);

        ROS_INFO("%d read done!  pose: (%.2lf, %.2lf, %.2lf), (w,x,y,z = (%.2lf,%.2lf,%.2lf,%.2lf))",
                    i,cur_pose.pose.position.x, cur_pose.pose.position.y, cur_pose.pose.position.z,
                      cur_pose.pose.orientation.w, cur_pose.pose.orientation.x, cur_pose.pose.orientation.y, cur_pose.pose.orientation.z
                );

        depth_data.stamp = time_stamp.toSec();
        depth_vec.push_back(depth_data);

        cur_pose.header.stamp = time_stamp;
        pub_cur_pose.publish(cur_pose);

        cv_bridge::CvImage out_msg;
        out_msg.header.stamp = time_stamp;
        out_msg.encoding = sensor_msgs::image_encodings::MONO8;
        out_msg.image = img.clone();
        pub_image.publish(out_msg.toImageMsg());

        getchar();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc,argv,"VaFRIC_warpper");
    ros::NodeHandle nh("~");

    pub_cur_pose = nh.advertise<geometry_msgs::PoseStamped>("cur_pose",1000);
    pub_image = nh.advertise<sensor_msgs::Image>("image",1000);
    pub_my_depth = nh.advertise<sensor_msgs::Image>("my_depth_visual",1000);
    pub_gt_depth = nh.advertise<sensor_msgs::Image>("gt_depth_visual",1000);
    pub_error_map = nh.advertise<sensor_msgs::Image>("error_map_visual",1000);
    ros::Subscriber sub_depth = nh.subscribe("/motion_stereo_left/depth/image_raw",1000,depth_callback);

    vafric_init();

    time_stamp = ros::Time::now();

    boost::thread th1(main_thread);
 
    while (ros::ok())
    {
        ros::spinOnce();
    }

    return 0;
}
