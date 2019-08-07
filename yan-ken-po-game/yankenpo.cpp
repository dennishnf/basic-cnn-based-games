#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sstream>


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



int main(int argc, char **argv)
{

  //! [Camera Preparation]  
  VideoCapture cam(0);
  Mat image;
  Mat image1;
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
  int counter=61;
  int va=0;
  int classId=0;
  string value="---";
  string value1="---";
  string winner="---";
   
  //! [Main Loop]
  for (;;)
  {
    //cv::TickMeter t;
    //t.start();
    cam.grab();
    cam.retrieve(image);
    resize(image, image, Size(), 1, 1);
    resize(image, image1, Size(), 1, 1);
    
    string s;
    stringstream out;
    out<<(counter/15);
    s=out.str();


    //! [Draw and insert text]
    if (s=="0") rectangle(image,Point(70,210),Point(280,420),Scalar(0,0,255),4);
    else rectangle(image,Point(70,210),Point(280,420),Scalar(255,0,0),4);
    if (s=="0") circle(image,Point(175,340),10,Scalar(0,0,255),2);
    else circle(image,Point(175,340),10,Scalar(255,0,0),2);
    
    putText(image, "counter: "+	s, Point(85,460), FONT_HERSHEY_DUPLEX, 0.9, Scalar(255,0,0),2,2);
    putText(image, "you", Point(205,78), FONT_HERSHEY_DUPLEX, 0.8, Scalar(0,0,0),2,2);
    putText(image, "pc", Point(100,78), FONT_HERSHEY_DUPLEX, 0.8, Scalar(0,0,0),2,2);
    putText(image, "winner: ", Point(120,35), FONT_HERSHEY_DUPLEX, 0.9, Scalar(0,255,0),2,2);
    //! [Draw and insert text]
    Mat small_image1;
    if (value1=="tijera"){ small_image1=imread("tijera.jpg");}
    if (value1=="piedra"){ small_image1=imread("piedra.jpg");}
    if (value1=="papel"){ small_image1=imread("papel.jpg");}
    if (value1=="---"){ small_image1=imread("none.jpg");}
    small_image1.copyTo(image(cv::Rect(70,90,small_image1.cols, small_image1.rows)));
    Mat small_image;
    if (value=="tijera"){ small_image=imread("tijera.jpg");}
    if (value=="piedra"){ small_image=imread("piedra.jpg");}
    if (value=="papel"){ small_image=imread("papel.jpg");}
    if (value=="---"){ small_image=imread("none.jpg");}
    small_image.copyTo(image(cv::Rect(180,90,small_image.cols, small_image.rows)));
    
    if (winner=="pc"){
      rectangle(image,Point(70,55),Point(170,190),Scalar(0,255,0),6);
      rectangle(image,Point(180,55),Point(280,190),Scalar(0,0,0),4);
    }
    else if (winner=="you"){
      rectangle(image,Point(70,55),Point(170,190),Scalar(0,0,0),4);
      rectangle(image,Point(180,55),Point(280,190),Scalar(0,255,0),6);
    }
    else {
      rectangle(image,Point(70,55),Point(170,190),Scalar(0,0,0),4);
      rectangle(image,Point(180,55),Point(280,190),Scalar(0,0,0),4);
    }
    
    imshow("Source", image);
    
    //extraction of region    
    Mat hand = image1(Rect(70,210,210,210));
   
    //! [Transformations]
    Mat bgr[3];
    split(hand,bgr);
    Mat thresh_r;
    Mat thresh_g;
    Mat thresh_b;
    threshold(bgr[2],thresh_r,80,255,THRESH_BINARY_INV);
    threshold(bgr[1],thresh_g,80,255,THRESH_BINARY_INV);
    threshold(bgr[0],thresh_b,80,255,THRESH_BINARY_INV);
    
    imshow("Hand", thresh_b);
   
    Mat hand_binary; 
    resize(thresh_b,hand_binary,Size(),0.23,0.23,INTER_CUBIC);
    //! [Transformations]
    
    int c = waitKey(10);
    
    //! [Recognize Object]
    if((char)c == 't') 
    {va=1;}
    if (va==1){
      
      //imwrite("hand.png",hand_binary);
      
      //select image from camera
      Mat img = hand_binary;
      //or select saved image
      //Mat img = imread("11.png");
      
      if (img.empty()){
        std::cerr << "Can't read image from the camera " << std::endl;
        exit(-1);
      }
      if (counter == 0)
      {
        //! [Prepare blob]
        //Model accepts only 48x48 one-channel images
        Mat inputBlob = blobFromImage(img, 1.0f, Size(48, 48),
                    Scalar(), false);   //Convert Mat to batch of images
        //! [Prepare blob]
        

        Mat prob;
        
        for (int i = 0; i < 1; i++)
        {
          //CV_TRACE_REGION("forward");
          //! [Set input blob]
          net.setInput(inputBlob);    //set the network input
          //! [Set input blob]
          
          //! [Make forward pass]
          prob = net.forward("prob");              //compute output
          //! [Make forward pass]
          
        }
        
        //! [Gather output]
        
        double classProb;
        getMaxClass(prob, &classId, &classProb);//find the best class
        //! [Gather output]
        
        if (classProb<=0.80){
          classId=3;
        }

        
        if (classId==0)
          value="piedra";
        else if(classId==1)
          value="papel";
        else if(classId==2)
          value="tijera";
        else if(classId==3)
          value="---";
        
        int v1 = rand() % 3;         // v1 in the range 0 to 2
        
        if (v1==0)
          value1="piedra";
        else if(v1==1)
          value1="papel";
        else if(v1==2)
          value1="tijera";

        //std::cout << value1 << std::endl;
        //std::cout << value << std::endl;
        
        
        if (value=="---"){
          winner="none";
        }
        else{
          if (value1=="piedra" and value=="piedra")
            winner="none";
          else if(value1=="piedra" and value=="papel")
            winner="you";
          else if(value1=="piedra" and value=="tijera")
            winner="pc";
          else if(value1=="papel" and value=="piedra")
            winner="pc";
          else if(value1=="papel" and value=="papel")
            winner="none";
          else if(value1=="papel" and value=="tijera")
            winner="you";
          else if(value1=="tijera" and value=="piedra")
            winner="you";
          else if(value1=="tijera" and value=="papel")
            winner="pc";
          else if(value1=="tijera" and value=="tijera")
            winner="none";
        }

        //std::cout << winner << std::endl;
        counter=61;
        va=0;
      }

      counter = counter -1;
            
    }
    //! [Recognize Object]
    
    
    if((char)c == 'q')
    { 
      break; 
    }

  }
  //! [Main Loop]
  
  cam.release();
  
}


