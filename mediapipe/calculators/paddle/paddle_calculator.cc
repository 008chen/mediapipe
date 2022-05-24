#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/calculators/paddle/paddle_calculator.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"

#include "mediapipe/logxx/testH.h"
#include <vector> 
namespace mediapipe
{
namespace {

constexpr char kImageTag[] = "IMAGE";
constexpr char kMaxCountTag[] = "MAXCOUNT";
}


  class PaddleCalculator : public CalculatorBase
  {

  public:
    PaddleCalculator(){};

    ~PaddleCalculator(){};

    static ::mediapipe::Status GetContract(CalculatorContract *cc)
    {
      cc->Inputs().Tag(kImageTag).Set<ImageFrame>();
      cc->Outputs().Tag(kImageTag).Set<ImageFrame>();
     
      if (cc->InputSidePackets().HasTag(kMaxCountTag)) {
        cc->InputSidePackets().Tag(kMaxCountTag).Set<int>();
      }
      return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status Open(CalculatorContext *cc)
    {
      const auto& options = cc->Options<::mediapipe::PaddleCalculatorOptions>();
      auto var = options.option_parameter_1();
      LOG(INFO) << "option_parameter_1:"<<var<<"xx:"<<vvprints();

      return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status Process(CalculatorContext *cc)
    {
      
      if(!cc->Inputs().Tag(kImageTag).IsEmpty())
      {
        // LOG(INFO) << data_stream;
        // std::unique_ptr<ImageFrame> output_stream = std::make_unique<ImageFrame>(cc->Inputs().Tag(kImageTag).Value());

        const auto& input_img = cc->Inputs().Tag(kImageTag).Get<ImageFrame>();
        cv::Mat input_mat = formats::MatView(&input_img);
        auto output_img = absl::make_unique<ImageFrame>(
        input_img.Format(), input_mat.cols, input_mat.rows);
        
        // LOG(INFO) << "opencv::"<<"input:"<<input_mat;
        cv::Mat output_mat = mediapipe::formats::MatView(output_img.get());
        // cv::Point p(200, 200);
        // cv::circle(input_mat, p, 100, cv::Scalar(0, 255, 0), -1);
        input_mat.copyTo(output_mat);


        cv::cvtColor(input_mat, input_mat, CV_BGRA2GRAY);
        // 利用 OTSU 二值化方法
        cv::threshold(input_mat, input_mat, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);
     

      //Define the erodent core 
        cv::Mat rec = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(50, 3));  
        //Etch binary image along horizontal, join text line 
        cv::Mat dilate0;  
        cv::erode(input_mat, dilate0, rec);  
        cv::Mat erode2;  
        //Image Reverse 
        cv::bitwise_not(dilate0, erode2);



        std::vector<cv::RotatedRect> rects;
        std::vector<std::vector<cv::Point>> counts;
        std::vector<cv::Vec4i> hierarchy;  
        //Extract the contour of the text line 
        cv::findContours(erode2, counts, hierarchy, CV_RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));



        //Mark the detection rectangles in the original image 
        for (int i = 0; i<counts.size(); i++)
        { 	
            //Culling of small contours 
          if (cv::contourArea(counts[i])<500) 			
                  continue; 	
            //Calculates the smallest rectangle with a vertical boundary for an contour 
            cv::Rect rect1 = cv::boundingRect(counts[i]); 
            // char ch[256];  		
              // cout << "hierarchy  num" << hierarchy[i] << endl << endl; 
                //Drawing Rectangular box 	
               cv::rectangle(output_mat, rect1, cv::Scalar(0, 0, 255), 1); 
        }


        // cc->Outputs()
        // .Tag(kImageTag)
        // .Add(output_img.release(), cc->InputTimestamp());
      cc->Outputs()
        .Tag(kImageTag)
        .Add(output_img.release(), cc->InputTimestamp());
      
        // cc->Outputs()
        // .Tag(kImageTag)
        // .AddPacket(cc->Inputs().Tag(kImageTag).Value());
      }

      return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status Close(CalculatorContext *cc)
    {

      return ::mediapipe::OkStatus();
    }
  };

  REGISTER_CALCULATOR(PaddleCalculator);

}