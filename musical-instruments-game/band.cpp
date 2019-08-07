#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sstream>

#include <iostream>
#include <thread>
#include <unistd.h>
#include <pthread.h>
#include <unistd.h>

using namespace std;
using namespace cv;

using namespace cv::dnn;

/* Find best class for the blob (i. e. class with maximal probability) */
static void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
  Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
  Point classNumber;
  
  minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
  *classId = classNumber.x;
}


//This function will be called from a thread

void call_from_thread0() {
//std::cout << "Hello, World" << std::endl;
system("mplayer 0.mp3 &");
}

void call_from_thread1() {
//std::cout << "Hello, World" << std::endl;
system("mplayer 1.mp3 &");
}

void call_from_thread2() {
//std::cout << "Hello, World" << std::endl;
system("mplayer 2.mp3 &");
}




int main(int argc, char **argv)
{

  //! [Camera Preparation]  
  VideoCapture cam(0);
  Mat image;
  Mat image1;

  //set camera params, CV_8UC1 grayscale, CV_8UC3 colored
  //cam.set(CV_CAP_PROP_FORMAT, CV_8UC3);

  
  const char szSourceWindow[] = "Source";
  namedWindow(szSourceWindow, WINDOW_AUTOSIZE);
  //! [Camera Preparation]

  //! [Neural Netowork Preparation]
  CV_TRACE_FUNCTION();

  String modelTxt = "model/model_deploy.prototxt";
  String modelBin = "model/train_iter_2000.caffemodel";
  
  Net net;
  try {
    //! [Read and initialize network]
    net = dnn::readNetFromCaffe(modelTxt, modelBin);
    //! [Read and initialize network]
  }
  catch (cv::Exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    //! [Check that network was read successfully]
    if (net.empty()){
      std::cerr << "Can't load network by using the following files: " << std::endl;
      std::cerr << "prototxt:   " << modelTxt << std::endl;
      std::cerr << "caffemodel: " << modelBin << std::endl;
      exit(-1);
    }
    //! [Check that network was read successfully]
  }
  //! [Neural Netowork Preparation]
  
  int classId=0;
  int classId_ant=0;
  int classId_act=0;
  string value="---";
   
  //! [Main Loop]
  for (;;)
  {
    classId_ant=classId;
    cam.grab();
    cam.retrieve(image);
    resize(image, image, Size(), 1, 1);
    resize(image, image1, Size(), 1, 1);
    
    //! [Draw and insert text]
    rectangle(image,Point(70,190),Point(280,400),Scalar(0,0,255),4);
    //circle(image,Point(175,320),10,Scalar(0,0,255),2);
    string s;
    stringstream out;
    out<<classId;
    s=out.str();
    //putText(image, value, Point(70,150), FONT_HERSHEY_DUPLEX, 1, Scalar(0,0,255),2,2);
    //! [Draw and insert text]
    
    imshow(szSourceWindow, image);
    
    //extraction of region    
    Mat hand = image1(Rect(70,190,210,210));
    
    //! [Transformations]
    Mat bgr[3];
    split(hand,bgr);
    Mat thresh_r;
    Mat thresh_g;
    Mat thresh_b;
    threshold(bgr[2],thresh_r,90,255,THRESH_BINARY_INV);
    threshold(bgr[1],thresh_g,90,255,THRESH_BINARY_INV);
    threshold(bgr[0],thresh_b,90,255,THRESH_BINARY_INV);
    
    imshow("Hand", thresh_b);
   
    Mat hand_binary; 
    resize(thresh_b,hand_binary,Size(),0.23,0.23,INTER_CUBIC);
    //! [Transformations]
    
    int c = waitKey(10);
    
    //! [Recognize Object]
    //if((char)c == 't') 
    //{
      
      //imwrite("hand.png",hand_binary);
      
      //select image from camera
      Mat img = hand_binary;
      //or select saved image
      //Mat img = imread("11.png");
      
      if (img.empty()){
        std::cerr << "Can't read image from the camera " << std::endl;
        exit(-1);
      }
      
      //! [Prepare blob]
      //Model accepts only 48x48 one-channel images
      Mat inputBlob = blobFromImage(img, 1.0f, Size(48, 48),
                  Scalar(), false);   //Convert Mat to batch of images
      //! [Prepare blob]
      
      Mat prob;
      cv::TickMeter t;
      for (int i = 0; i < 1; i++)
      {
        //CV_TRACE_REGION("forward");
        //! [Set input blob]
        net.setInput(inputBlob);    //set the network input
        //! [Set input blob]
        t.start();
        //! [Make forward pass]
        prob = net.forward("prob");              //compute output
        //! [Make forward pass]
        t.stop();
      }
      
      //! [Gather output]
      
      double classProb;
      getMaxClass(prob, &classId, &classProb);//find the best class

      if (classProb<=0.80){
        classId=2;
      }
      //! [Gather output]
      
     if (classId==0)
       value="PIANO";
     else if(classId==1)
       value="BATERIA";
     else if(classId==2)
       value="APLAUSOS";
    



      //! [Print results]
      //std::cout << "\n========================================================" << std::endl;
      //std::cout << "Best class: #" << classId << std::endl;
      //std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
      //! [Print results]
      std::cout << "Time: " << (double)t.getTimeMilli() / t.getCounter() << " ms (average from " << t.getCounter() << " iterations)" << std::endl;
      //resize(img, img, Size(), 0.75, 0.75);
      //imshow("image", img);

      classId_act=classId;      
      

      if (classId_ant==0 and classId_act==0){
        //system("pkill mplayer");
        //std::thread t1(call_from_thread0);
        //t1.join();
      }

      if (classId_ant==0 and classId_act==1){
        system("pkill mplayer");
        std::thread t1(call_from_thread1);
        t1.join();
      }

      if (classId_ant==0 and classId_act==2){
        system("pkill mplayer");
        std::thread t1(call_from_thread2);
        t1.join();
      }

      if (classId_ant==1 and classId_act==0){
        system("pkill mplayer");
        std::thread t1(call_from_thread0);
        t1.join();
      }

      if (classId_ant==1 and classId_act==1){
        //system("pkill mplayer");
        //std::thread t1(call_from_thread1);
        //t1.join();
      }

      if (classId_ant==1 and classId_act==2){
        system("pkill mplayer");
        std::thread t1(call_from_thread2);
        t1.join();
      }

      if (classId_ant==2 and classId_act==0){
        system("pkill mplayer");
        std::thread t1(call_from_thread0);
        t1.join();
      }

      if (classId_ant==2 and classId_act==1){
        system("pkill mplayer");
        std::thread t1(call_from_thread1);
        t1.join();
      }

      if (classId_ant==2 and classId_act==2){
        //system("pkill mplayer");
        //std::thread t1(call_from_thread2);
        //t1.join();
      }


    //}
    //! [Recognize Object]
    
    if((char)c == 'q')
    {
      system("pkill mplayer");
      break; 
    }

  }
  //! [Main Loop]
  
  cam.release();
  
}




