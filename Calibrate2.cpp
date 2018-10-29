#include "opencv2/opencv.hpp"
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <experimental/filesystem>
namespace fs =  std::experimental::filesystem::v1;

using namespace cv;

// Read the lidar points from file into point array
void read_file(std::ifstream &file, std::vector<cv::Point3d> &point_array){
  std::string str;
  while (std::getline(file, str) )
  {
    std::string data = str.substr(str.find(':')+1, str.find(']')-1);
    while(data.length() > 1){

      std::string entry = data.substr(0,data.find(';'));
      
      double d = std::stof(entry.substr(0,entry.find('|')));
      entry.erase(0, entry.find('|')+1);
      double h = std::stof(entry.substr(0,entry.find('|')));
      entry.erase(0, entry.find('|')+1);
      double v = std::stof(entry.substr(0,entry.find('|')));

      data.erase(0, data.find(';')+1);
      
      double diag = d * cos(v);

      point_array.push_back( Point3d( diag * cos(h), diag * sin(h), d * -sin(v) ) );
    }
  }
};


// Calculates rotation matrix given euler angles.
cv::Mat eulerAnglesToRotationMatrix(double* theta)
{
    // Calculate rotation about x axis
    cv::Mat R_x = (cv::Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );
     
    // Calculate rotation about y axis
    cv::Mat R_y = (Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );
     
    // Calculate rotation about z axis
    cv::Mat R_z = (Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);
     
     
    // Combined rotation matrix
    cv::Mat R = R_z * R_y * R_x;
     
    return R;
 
}


int main(int, char**)
{
  //Rotation vector of camera
  double rv [3] = {0., 0., 0.};
  rv[0] = 0.;//0.;//-1.3;
  rv[1] = -0.018;//-0.02;//-2.0;
  rv[2] = 0.02;//0.018;

  std::vector<cv::Point2d> imagePoints;
  std::vector<cv::Point3d> objectPoints;

  //Focus distance and resolution of video
  cv::Mat K (3,3,cv::DataType<double>::type);
  K.at<double>(0,0) = 1000; 
  K.at<double>(0,1) = 0.;
  K.at<double>(0,2) = 640.;//960.;

  K.at<double>(1,0) = 0.;
  K.at<double>(1,1) = 1000;
  K.at<double>(1,2) = 360;//540;

  K.at<double>(2,0) = 0.;
  K.at<double>(2,1) = 0.;
  K.at<double>(2,2) = 1.;

  //Intrinsic distortion coefficeints
  cv::Mat distCoeffs(5,1,cv::DataType<double>::type);
  distCoeffs.at<double>(0) = 0.;//-4.5543230203932972e-01;
  distCoeffs.at<double>(1) = 0.;//5.7444032014766666e-01;
  distCoeffs.at<double>(2) = 0.;
  distCoeffs.at<double>(3) = 0.;
  distCoeffs.at<double>(4) = 0.;//-1.3972235806313755e+00;

  //Default rotation vector between camera coordinates and lidar coordinates
  cv::Mat rvecR(3,3,cv::DataType<double>::type);
  cv::Mat baseR(3,3,cv::DataType<double>::type);
  baseR.at<double>(0,0) = 0.;
  baseR.at<double>(1,0) = 0.;
  baseR.at<double>(2,0) = 1.;
  baseR.at<double>(0,1) = -1.;
  baseR.at<double>(1,1) = 0.;
  baseR.at<double>(2,1) = 0.;
  baseR.at<double>(0,2) = 0.;
  baseR.at<double>(1,2) = -1.;
  baseR.at<double>(2,2) = 0.;
  //Transform base camera rotation from vector to matrix format
  cv::Mat er = eulerAnglesToRotationMatrix(rv);
  //Rotation vector: camera rotation * default rotation
  rvecR = baseR*er;

  cv::Mat rvec(3,1,cv::DataType<double>::type);

  //Transform back to readable vector format
  cv::Rodrigues(rvecR,rvec);

  std::cout<<std::endl;
  std::cout<<rvec.at<double>(0)<<":"<<rvec.at<double>(1)<<":"<<rvec.at<double>(2)<<std::endl;

  //Translation vector between camera and lidar
  cv::Mat tvec(3,1,cv::DataType<double>::type);
  tvec.at<double>(0) = 0.0;
  tvec.at<double>(1) = 0.0;
  tvec.at<double>(2) = 0.0;

  //read data from files into variables
  Mat image;
  std::ifstream file("./probe_points.txt");
  Mat baseImage = imread("./probe.png", CV_LOAD_IMAGE_COLOR); 
  std::vector<cv::Point3d> points;
  read_file(file, points);
    
  std::vector<cv::Point2d> projection_points;

  namedWindow( "Projection Calibration",WINDOW_AUTOSIZE);

  double astep = 0.002;
  double tstep = 0.05;
  //Check that data was read correctly
  if(! baseImage.data ){
    // Check for invalid input
    std::cout <<  "Could not open or find the image" << std::endl ;
    return -1;
  }
    
  for(;;){
    //Read refresh the frame by reading values anew and reprojecting points
    baseImage.copyTo(image);
    std::cout<<"new step: "<<std::endl;  
    std::cout<<"tvec: "<<tvec.at<double>(0)<<":"<<tvec.at<double>(1)<<":"<<tvec.at<double>(2)<<std::endl;
    std::cout<<"rvecE: "<<rv[0]<<":"<<rv[1]<<":"<<rv[2]<<std::endl;
    std::cout<<"focus: "<<K.at<double>(0,0)<<std::endl;
	
    cv::Mat er = eulerAnglesToRotationMatrix(rv);
    rvecR = baseR*er;
    cv::Rodrigues(rvecR,rvec);

    std::cout<<"rvec: "<<rvec.at<double>(0)<<":"<<rvec.at<double>(1)<<":"<<rvec.at<double>(2)<<std::endl;

    projection_points.clear();
    
    projectPoints( points, rvec, tvec, K, distCoeffs, projection_points );

    //Draw projected points
    for(int i=0; i < projection_points.size(); i++){
      if(projection_points[i].x < 0 || projection_points[i].x > 1280 || projection_points[i].y < 0 || projection_points[i].y > 720){
        //std::cout<<projection_points[i].x<<":"<<projection_points[i].y<<std::endl;
	continue;
      }
      int color = points[i].x*255/7;
      if(color > 255){ color = 255;}
	  circle(image, projection_points[i], 2, Scalar( color,0,255-color ),2);
      }
	
      //Show image on screen
      imshow( "Projection Calibration", image );                   

      //Receive command from user to adjust the error matrices and reproject points
      int waitingForKey = 1;
      while(waitingForKey){
        int key = waitKey(30);                                        // Wait for a keystroke in the window
	switch(key){
	  case 'w':
	    waitingForKey = 0;
	    tvec.at<double>(1) -= tstep; 
	    break;
	  case 'a':
	    waitingForKey = 0;
	    tvec.at<double>(0) -= tstep;
	    break;
	  case 's':
	    waitingForKey = 0;
	    tvec.at<double>(1) += tstep;
            break;
	  case 'd':
	    waitingForKey = 0;
	    tvec.at<double>(0) += tstep;
	    break;
	  case 'r':
	    waitingForKey = 0;
	    tvec.at<double>(2) += tstep;
	    break;
	  case 'f':
	    waitingForKey = 0;
	    tvec.at<double>(2) -= tstep;
	    break;
	  case 'u':
	    waitingForKey = 0;
	    rv[1] += astep; 
	    break;
	  case 'h':
	    waitingForKey = 0;
	    rv[2] -= astep;
	    break;
	  case 'j':
	    waitingForKey = 0;
	    rv[1] -= astep;
	    break;
	  case 'k':
	    waitingForKey = 0;
	    rv[2] += astep;
	    break;
	  case 'o':
	    waitingForKey = 0;
	    rv[0] += astep;
	    break;
	  case 'l':
	    waitingForKey = 0;
	    rv[0] -= astep;
	    break;
	  case 'z':
	    waitingForKey = 0;
	    K.at<double>(0,0) += 50;
	    K.at<double>(1,1) += 50;
	    break;
	  case 'x':
	    waitingForKey = 0;
	    K.at<double>(0,0) -= 50;
	    K.at<double>(1,1) -= 50;
	    break;
	  case '+':
	    waitingForKey = 0;
	    rv[0] = 0.;
  	    rv[1] = 0.;
  	    rv[2] = 0.;
	    break;
	  case 27:
      	    return 0;
	}
	if(rv[0]<astep && rv[0]>-astep){rv[0]=0.;}
	if(rv[1]<astep && rv[1]>-astep){rv[1]=0.;}
	if(rv[2]<astep && rv[2]>-astep){rv[2]=0.;}
	if(tvec.at<double>(0)<tstep && tvec.at<double>(0)>-tstep){tvec.at<double>(0)=0.;}
	if(tvec.at<double>(1)<tstep && tvec.at<double>(1)>-tstep){tvec.at<double>(1)=0.;}
	if(tvec.at<double>(2)<tstep && tvec.at<double>(2)>-tstep){tvec.at<double>(2)=0.;}
      }
    }
  return 0;
}



