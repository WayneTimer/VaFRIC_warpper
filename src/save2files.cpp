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
ros::Time time_stamp;
double translation_square_diff = 1.0;

int file_cnt = 0;

void vafric_init()
{
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
        z /= 100.0;  // vafric dataset use cm, /100.0 to m

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

    pose.position.x = gt_p(0) / 100.0;    // vafric dataset use cm, /100.0 to m
    pose.position.y = gt_p(1) / 100.0;    // vafric dataset use cm, /100.0 to m
    pose.position.z = gt_p(2) / 100.0;    // vafric dataset use cm, /100.0 to m
    pose.orientation.w = gt_q.w();
    pose.orientation.x = gt_q.x();
    pose.orientation.y = gt_q.y();
    pose.orientation.z = gt_q.z();    

    stamp = stamp + ros::Duration(1.0/HZ);
}

void main_thread()
{
    bool first_pose = true;
    geometry_msgs::PoseStamped last_pose;

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

        if (!first_pose)
        {
            // judge new frame translatioin diff
            Eigen::Vector3d lTr; // T_r^l
            Eigen::Vector3d last_T,cur_T;
            Eigen::Matrix3d last_R;

            last_R = Eigen::Quaterniond{last_pose.pose.orientation.w,
                                        last_pose.pose.orientation.x,
                                        last_pose.pose.orientation.y,
                                        last_pose.pose.orientation.z}.toRotationMatrix();
            last_T = Eigen::Vector3d(last_pose.pose.position.x,last_pose.pose.position.y,last_pose.pose.position.z);
            cur_T = Eigen::Vector3d(cur_pose.pose.position.x,cur_pose.pose.position.y,cur_pose.pose.position.z);
            lTr = last_R.transpose() * (cur_T - last_T);

            translation_square_diff = lTr(0)*lTr(0) + lTr(1)*lTr(1) + lTr(2)*lTr(2);
            ROS_WARN("translation_square_diff = %lf",translation_square_diff);
        }

        char file_name[100];

        // save images
        sprintf(file_name,"/home/timer/work_git/Teddy/vafric_results/%04d.png",file_cnt);
        cv::imwrite(file_name,img);

        // save depth
        sprintf(file_name,"/home/timer/work_git/Teddy/vafric_results/%04d.pgm",file_cnt);
        FILE *depth_file = fopen(file_name,"w");
        for (int v=0;v<cali.height[0];v++)
            for (int u=0;u<cali.width[0];u++)
            {
                fprintf(depth_file,"%lf ",depth_data.depth(v,u));
            }
        fclose(depth_file);

        // save pose
        sprintf(file_name,"/home/timer/work_git/Teddy/vafric_results/%04d.txt",file_cnt);
        FILE *pose_file = fopen(file_name,"w");
        Eigen::Quaterniond quat;
        quat.w() = cur_pose.pose.orientation.w;
        quat.x() = cur_pose.pose.orientation.x;
        quat.y() = cur_pose.pose.orientation.y;
        quat.z() = cur_pose.pose.orientation.z;
        Eigen::Matrix3d rotation = quat.toRotationMatrix();

        for (int i=0;i<3;i++)
        {
            for (int j=0;j<3;j++)
                fprintf(pose_file,"%lf ", rotation(i,j));
            fprintf(pose_file,"\n");
        }
        fprintf(pose_file,"%lf %lf %lf\n",cur_pose.pose.position.x,cur_pose.pose.position.y,cur_pose.pose.position.z);
        fclose(pose_file);


        file_cnt++;

        last_pose = cur_pose;
        first_pose = false;

        getchar();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc,argv,"save2files");
    ros::NodeHandle nh("~");

    vafric_init();
    main_thread();

    return 0;
}
